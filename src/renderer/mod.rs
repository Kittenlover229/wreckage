use std::{borrow::Borrow, cell::RefCell, sync::Arc};

use egui_winit_vulkano::Gui;
use log::debug;
use nalgebra_glm::{Mat4, Quat, Vec3, vec3};
use smallvec::{smallvec, SmallVec};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BlitImageInfo,
        CommandBufferUsage, CopyImageToBufferInfo, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageCreateFlags, ImageUsage, StorageImage, SwapchainImage,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    pipeline::{ComputePipeline, Pipeline},
    swapchain::{
        self, AcquireError, PresentMode, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};

use self::object::{MaterialTableBuffer, SphereBufferData, BVHAABB};

mod object;
mod shaders;

#[derive(Clone, Debug, Default)]
#[repr(C)]
pub struct DynamicCameraData {
    pub pos: Vec3,
    pub rotation: Quat,
    // Vertical field of view
    pub fov: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    // TODO: Maybe move out of dynamic data?
    pub samples: u32,
}

impl DynamicCameraData {
    pub fn view_matrix(&self) -> Mat4 {
        nalgebra_glm::quat_to_mat4(&self.rotation)
    }
}

pub struct Camera {
    pub downscale_factor: u32,
    pub idx: u32,
    pub width: u32,
    pub height: u32,

    pub dynamic_data: RefCell<DynamicCameraData>,

    pub descriptors: Arc<PersistentDescriptorSet>,
    pub out_buffer: Arc<dyn ImageAccess>,
    pub data_buffer: Subbuffer<CameraDataBuffer>,
    pub render_command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

impl Camera {
    pub fn refresh_data_buffer(&mut self) -> anyhow::Result<()> {
        let mut buf = self.data_buffer.write()?;
        let dynamic_data = self.dynamic_data.borrow();
        buf.fov = dynamic_data.fov;
        buf.aspect_ratio = self.width as f32 / self.height as f32;
        buf.view_matrix = dynamic_data.view_matrix().data.0;
        buf.near_plane = self.dynamic_data.borrow().near_plane;
        buf.far_plane = self.dynamic_data.borrow().far_plane;
        Ok(())
    }

    pub fn pixel_area(&self) -> u32 {
        self.width * self.height
    }
}

pub struct SwapchainPresenter {
    pub camera_idx: u32,
    pub dirty: bool,
    pub idx: u32,

    pub swapchain: Arc<Swapchain>,
    pub images: SmallVec<[Arc<SwapchainImage>; 16]>,
    pub blit_command_buffers: SmallVec<[Arc<PrimaryAutoCommandBuffer>; 4]>,
}

#[derive(BufferContents, Debug, Default)]
#[repr(C)]
pub struct CameraDataBuffer {
    pub view_matrix: [[f32; 4]; 4],
    pub origin_offset: [[f32; 3]; 1],
    pub aspect_ratio: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub fov: f32,
}

// Buffer for data updated every frame
#[derive(BufferContents, Debug)]
#[repr(C)]
pub struct HotBuffer {
    time: u32,
}

#[derive(BufferContents)]
#[repr(C)]
pub struct ReadonlyConstants {
    pub material_table: MaterialTableBuffer,
}

pub struct Renderer {
    pub cameras: SmallVec<[Arc<RefCell<Camera>>; 8]>,
    pub swapchains: SmallVec<[SwapchainPresenter; 1]>,

    // Allocators
    pub buffer_allocator: StandardMemoryAllocator,
    pub command_allocator: StandardCommandBufferAllocator,
    pub descriptor_allocator: StandardDescriptorSetAllocator,

    // State Management
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub physical: Arc<PhysicalDevice>,
    pub fallback_queue: Arc<Queue>,

    // Rendering
    pub compute_pipeline: Arc<ComputePipeline>,
    pub readonly_constants: Subbuffer<ReadonlyConstants>,
    pub hotbuffer: Subbuffer<HotBuffer>,

    // Objects
    pub objects: (Subbuffer<[BVHAABB]>, Subbuffer<[SphereBufferData]>),
}

impl Renderer {
    pub fn camera_format() -> Format {
        Format::R8G8B8A8_UNORM
    }

    pub fn new(library: Arc<VulkanLibrary>) -> anyhow::Result<Self> {
        let required_extensions = vulkano_win::required_extensions(&library);

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )?;

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let physical_device = instance
            .enumerate_physical_devices()?
            .min_by_key(|p| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,

                // Note that there exists `PhysicalDeviceType::Other`, however,
                // `PhysicalDeviceType` is a non-exhaustive enum. Thus, one should
                // match wildcard `_` to catch all unknown device types.
                _ => 4,
            })
            .unwrap();

        debug!("Using device {}", physical_device.properties().device_name);

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::COMPUTE | QueueFlags::TRANSFER | QueueFlags::GRAPHICS)
            })
            .expect("couldn't find a graphical queue family")
            as u32;

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                // here we pass the desired queue family to use by index
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )?;

        let queue = queues.next().unwrap();

        let buffer_allocator = StandardMemoryAllocator::new_default(device.clone());
        let command_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        let descriptor_allocator = StandardDescriptorSetAllocator::new(device.clone());
        let shader = shaders::main::load(device.clone())?;
        let pipeline = ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )?;

        let readonly_constants = Buffer::from_data(
            &buffer_allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            ReadonlyConstants {
                material_table: MaterialTableBuffer::default(),
            },
        )?;

        let hotbuffer = Buffer::from_data(
            &buffer_allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            HotBuffer { time: 0u32 },
        )?;

        let mut bvhes = [BVHAABB::default(); 4096];
        let mut spheres = [SphereBufferData::default(); 4096];

        spheres[0].center = vec3(0., 0., 1.).data.0[0];
        spheres[0].radius = 0.1f32;
        bvhes[0].aabb_center = spheres[0].center;
        bvhes[0].aabb_max = (vec3(0., 0., 1.) + vec3(0.1, 0.1, 0.1)).data.0[0];
        bvhes[0].aabb_min = (vec3(0., 0., 1.) - vec3(0.1, 0.1, 0.1)).data.0[0];
        bvhes[0].left_idx = 0;
        bvhes[0].right_idx = 0;
        bvhes[0].object_id = 1;

        let bvhes = Buffer::from_iter(
            &buffer_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            bvhes,
        )?;

        let spheres = Buffer::from_iter(
            &buffer_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            spheres,
        )?;

        Ok(Self {
            objects: (bvhes, spheres),
            hotbuffer,
            cameras: smallvec![],
            swapchains: smallvec![],
            instance,
            device,
            fallback_queue: queue,
            physical: physical_device,
            buffer_allocator,
            command_allocator,
            descriptor_allocator,
            compute_pipeline: pipeline,
            readonly_constants,
        })
    }

    #[must_use]
    pub fn attach_swapchain(
        &mut self,
        camera_idx: u32,
        surface: Arc<Surface>,
    ) -> anyhow::Result<u32> {
        let caps = self
            .physical
            .surface_capabilities(&surface, Default::default())?;

        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = self
            .physical
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        let (swapchain, images) = Swapchain::new(
            self.device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count + 1,
                image_format: Some(image_format),
                image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                composite_alpha,
                // TODO: check if present mode is supported
                present_mode: PresentMode::Immediate,
                ..Default::default()
            },
        )?;

        let mut blit_command_buffers = smallvec![];
        for image in &images {
            let mut builder = AutoCommandBufferBuilder::primary(
                &self.command_allocator,
                self.fallback_queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )?;

            let camera: &RefCell<Camera> = self.cameras[camera_idx as usize].borrow();
            let camera = camera.borrow();

            builder.blit_image(BlitImageInfo::images(
                camera.out_buffer.clone(),
                image.clone(),
            ))?;
            let command_buffer = builder.build()?;

            blit_command_buffers.push(Arc::new(command_buffer));
        }

        let swapchain_preseneter = SwapchainPresenter {
            dirty: false,
            idx: self.swapchains.len() as u32,
            swapchain,
            images: SmallVec::from_vec(images),
            camera_idx,
            blit_command_buffers,
        };

        let idx = swapchain_preseneter.idx;
        self.swapchains.push(swapchain_preseneter);
        Ok(idx)
    }

    pub fn refresh_swapchain(&mut self, idx: u32, dimensions: [u32; 2]) -> anyhow::Result<()> {
        let s = &mut self.swapchains[idx as usize];
        let (swapchain, images) = match s.swapchain.recreate(SwapchainCreateInfo {
            image_extent: dimensions,
            ..s.swapchain.create_info()
        }) {
            Ok(k) => k,
            Err(swapchain::SwapchainCreationError::ImageExtentNotSupported { .. }) => return Ok(()),
            Err(e) => return Err(e.into()),
        };

        let mut camera = self.cameras[s.camera_idx as usize].as_ref().borrow_mut();
        let samples = camera.dynamic_data.borrow().samples.to_owned();

        let mut cmd_builder = AutoCommandBufferBuilder::primary(
            &self.command_allocator,
            self.fallback_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )?;

        camera.width = dimensions[0];
        camera.height = dimensions[1];

        camera.out_buffer = StorageImage::with_usage(
            &self.buffer_allocator,
            vulkano::image::ImageDimensions::Dim2d {
                width: camera.width / camera.downscale_factor,
                height: camera.height / camera.downscale_factor,
                array_layers: samples,
            },
            Renderer::camera_format(),
            ImageUsage::TRANSFER_SRC
                | ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::SAMPLED
                | ImageUsage::STORAGE,
            ImageCreateFlags::MUTABLE_FORMAT,
            [self.fallback_queue.queue_family_index()],
        )?;

        let pipeline_layout = self.compute_pipeline.layout();
        let descriptor_layouts = pipeline_layout.set_layouts();

        camera.descriptors = PersistentDescriptorSet::new(
            &self.descriptor_allocator,
            descriptor_layouts[0].clone(),
            // TODO: dry this
            [
                WriteDescriptorSet::image_view(
                    0,
                    ImageView::new_default(camera.out_buffer.clone())?,
                ),
                WriteDescriptorSet::buffer(1, camera.data_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.readonly_constants.clone()),
                WriteDescriptorSet::buffer(3, self.hotbuffer.clone()),
                WriteDescriptorSet::buffer(4, self.objects.0.clone()),
                WriteDescriptorSet::buffer(5, self.objects.1.clone()),
            ],
        )?;

        cmd_builder
            .bind_descriptor_sets(
                self.compute_pipeline.bind_point(),
                self.compute_pipeline.layout().clone(),
                0,
                camera.descriptors.clone(),
            )
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .dispatch([
                camera.width / camera.downscale_factor,
                camera.height / camera.downscale_factor,
                camera.dynamic_data.borrow().samples,
            ])?;
        camera.render_command_buffer = Arc::new(cmd_builder.build()?);
        camera.refresh_data_buffer()?;

        let mut blit_command_buffers = smallvec![];
        for image in &images {
            let mut builder = AutoCommandBufferBuilder::primary(
                &self.command_allocator,
                self.fallback_queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )?;

            builder.blit_image(BlitImageInfo::images(
                camera.out_buffer.clone(),
                image.clone(),
            ))?;

            blit_command_buffers.push(Arc::new(builder.build()?));
        }

        s.blit_command_buffers = blit_command_buffers;
        s.dirty = false;
        s.swapchain = swapchain;
        s.images = SmallVec::from_vec(images);

        Ok(())
    }

    pub fn add_camera(
        &mut self,
        dynamic_data: DynamicCameraData,
        downscale_factor: u32,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Arc<RefCell<Camera>>> {
        let dynamic_data = dynamic_data.borrow();
        let out_buffer = StorageImage::with_usage(
            &self.buffer_allocator,
            vulkano::image::ImageDimensions::Dim2d {
                width: width / downscale_factor,
                height: height / downscale_factor,
                array_layers: dynamic_data.samples,
            },
            Renderer::camera_format(),
            ImageUsage::TRANSFER_SRC
                | ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::SAMPLED
                | ImageUsage::STORAGE,
            ImageCreateFlags::MUTABLE_FORMAT,
            [self.fallback_queue.queue_family_index()],
        )?;

        let camera_data_buffer = Buffer::from_data(
            &self.buffer_allocator,
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            CameraDataBuffer {
                view_matrix: dynamic_data.view_matrix().data.0,
                origin_offset: dynamic_data.pos.data.0,
                aspect_ratio: width as f32 / height as f32,
                near_plane: dynamic_data.near_plane,
                far_plane: dynamic_data.far_plane,
                fov: dynamic_data.fov,
            },
        )?;

        let pipeline_layout = self.compute_pipeline.layout();
        let descriptor_layouts = pipeline_layout.set_layouts();

        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_allocator,
            descriptor_layouts[0].clone(),
            // TODO: dry this
            [
                WriteDescriptorSet::image_view(
                    0,
                    // possibly redundant
                    ImageView::new_default(out_buffer.clone())?,
                ),
                WriteDescriptorSet::buffer(1, camera_data_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.readonly_constants.clone()),
                WriteDescriptorSet::buffer(3, self.hotbuffer.clone()),
                WriteDescriptorSet::buffer(4, self.objects.0.clone()),
                WriteDescriptorSet::buffer(5, self.objects.1.clone()),
            ],
        )?;

        // TODO: DRY THIS
        let mut cmd_builder = AutoCommandBufferBuilder::primary(
            &self.command_allocator,
            self.fallback_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )?;

        cmd_builder
            .bind_descriptor_sets(
                self.compute_pipeline.bind_point(),
                self.compute_pipeline.layout().clone(),
                0,
                descriptor_set.clone(),
            )
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .dispatch([
                width / downscale_factor,
                height / downscale_factor,
                dynamic_data.borrow().samples,
            ])?;

        let camera = Arc::new(RefCell::new(Camera {
            width,
            height,
            dynamic_data: RefCell::new(dynamic_data.to_owned()),
            out_buffer,
            idx: self.cameras.len() as u32,
            descriptors: descriptor_set,
            downscale_factor,
            data_buffer: camera_data_buffer,
            render_command_buffer: Arc::new(cmd_builder.build()?),
        }));

        self.cameras.push(camera.clone());

        Ok(camera)
    }

    pub fn draw_all(&self) -> anyhow::Result<Box<dyn GpuFuture>> {
        {
            let mut hotbuffer = self.hotbuffer.write()?;
            hotbuffer.time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .subsec_micros();
        }

        for camera in &self.cameras {
            let camera: &RefCell<Camera> = camera.as_ref();
            let camera = camera.borrow();
            let future = sync::now(self.device.clone()).then_execute(
                self.fallback_queue.clone(),
                camera.render_command_buffer.clone(),
            )?;
            return Ok(Box::new(future));
        }

        unimplemented!();
    }

    pub fn present(
        &mut self,
        swapchain_index: u32,
        before: Box<dyn GpuFuture>,
        gui: &mut Gui,
    ) -> anyhow::Result<()> {
        let SwapchainPresenter {
            dirty, camera_idx, ..
        } = self.swapchains[swapchain_index as usize];
        if dirty {
            let cam = self.cameras[camera_idx as usize].as_ref().borrow();
            let width = cam.width;
            let height = cam.height;
            let downscale_factor = cam.downscale_factor;
            drop(cam);

            self.refresh_swapchain(
                swapchain_index,
                [width / downscale_factor, height / downscale_factor],
            )?;
        }

        let s = &mut self.swapchains[swapchain_index as usize];
        let SwapchainPresenter {
            swapchain,
            dirty,
            images,
            blit_command_buffers,
            ..
        } = s;

        if *dirty {
            return Ok(());
        }

        let (image_i, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    *dirty = true;
                    return Ok(());
                }
                Err(e) => return Err(e.into()),
            };

        if suboptimal {
            *dirty = true;
        }

        let execution = sync::now(self.device.clone())
            .join(before)
            .join(acquire_future)
            .then_execute(
                self.fallback_queue.clone(),
                blit_command_buffers[image_i as usize].clone(),
            )?
            .then_signal_fence();

        let execution = gui
            .draw_on_image(
                execution,
                ImageView::new_default(images[image_i as usize].clone())?,
            )
            .then_swapchain_present(
                self.fallback_queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
            )
            .then_signal_fence_and_flush();

        match execution {
            Ok(future) => future.wait(None)?,
            Err(FlushError::OutOfDate) => *dirty = true,
            Err(e) => return Err(e.into()),
        }

        Ok(())
    }

    pub fn save_png(&self, name: &str, camera_idx: u32) -> anyhow::Result<()> {
        let camera: &RefCell<Camera> = self.cameras[camera_idx as usize].borrow();
        let camera = camera.borrow();

        let buf = Buffer::from_iter(
            &self.buffer_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Download,
                ..Default::default()
            },
            (0..camera.pixel_area() * 4 * camera.dynamic_data.borrow().samples).map(|_| 0u8),
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_allocator,
            self.fallback_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            camera.out_buffer.clone(),
            buf.clone(),
        ))?;

        let command_buffer = builder.build()?;

        sync::now(self.device.clone())
            .then_execute(self.fallback_queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        use image::{ImageBuffer, Rgba};

        let buffer_content = buf.read()?;

        let buf: Vec<u8> = buffer_content
            .iter()
            .enumerate()
            // Ignore the alpha channel since it's used for depth
            .map(|(i, b)| if (i + 1) % 4 == 0 { std::u8::MAX } else { *b })
            .take(4 * camera.pixel_area() as usize)
            .collect();

        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
            camera.width / camera.downscale_factor,
            camera.height / camera.downscale_factor,
            buf,
        )
        .unwrap();
        image.save(name)?;

        Ok(())
    }
}

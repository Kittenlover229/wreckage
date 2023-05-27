use std::{cell::RefCell, sync::Arc};

use nalgebra_glm::{Quat, Vec3};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BlitImageInfo,
        ClearColorImageInfo, CommandBufferUsage, CopyImageToBufferInfo,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
    },
    format::ClearColorValue,
    image::{ImageAccess, ImageUsage, StorageImage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    swapchain::{
        self, AcquireError, PresentInfo, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    VulkanLibrary,
};

#[derive(Debug, Clone)]
pub enum RenderableObjectKind {
    Sphere { radius: f32 },
}

#[derive(Debug, Clone)]
pub struct RenderableObject {
    pub kind: RenderableObjectKind,
    pub pos: Vec3,
    pub rotation: Quat,
}

#[derive(Clone, Debug, Default)]
pub struct CameraOptions {
    pub pos: Vec3,
    pub rotation: Quat,
    pub fov: f32,
}

pub struct Camera {
    pub idx: u32,
    pub width: u32,
    pub height: u32,

    pub options: RefCell<CameraOptions>,

    pub out_buffer: Arc<dyn ImageAccess>,
}

pub struct SwapchainPresenter {
    pub camera_idx: u32,
    pub dirty: bool,
    pub idx: u32,

    pub swapchain: Arc<Swapchain>,
    pub images: Vec<Arc<SwapchainImage>>,
}

pub struct Renderer {
    pub cameras: Vec<Arc<Camera>>,
    swapchains: Vec<SwapchainPresenter>,

    // Allocators
    pub buffer_allocator: StandardMemoryAllocator,
    pub command_allocator: StandardCommandBufferAllocator,

    // State Management
    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub physical: Arc<PhysicalDevice>,
    pub fallback_queue: Arc<Queue>,
}

impl Renderer {
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
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .next()
            .unwrap();

        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(QueueFlags::COMPUTE | QueueFlags::TRANSFER)
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

        Ok(Self {
            cameras: vec![],
            swapchains: vec![],
            instance,
            device,
            fallback_queue: queue,
            physical: physical_device,
            buffer_allocator,
            command_allocator,
        })
    }

    #[must_use]
    pub fn attach_swapchain(&mut self, camera_idx: u32, surface: Arc<Surface>) -> u32 {
        let camera = self.cameras[camera_idx as usize].clone();

        let caps = self
            .physical
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");

        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = Some(
            self.physical
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        let (swapchain, images) = Swapchain::new(
            self.device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count + 1,
                image_format,
                image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap();

        let swapchain_preseneter = SwapchainPresenter {
            dirty: false,
            idx: self.swapchains.len() as u32,
            swapchain,
            images,
            camera_idx,
        };

        let idx = swapchain_preseneter.idx;
        self.swapchains.push(swapchain_preseneter);
        idx
    }

    pub fn refresh_swapchain(&mut self, idx: u32, dimensions: [u32; 2]) -> anyhow::Result<()> {
        let s = &mut self.swapchains[idx as usize];
        let (swapchain, images) = match s.swapchain.recreate(SwapchainCreateInfo {
            image_extent: dimensions.into(),
            ..s.swapchain.create_info()
        }) {
            Ok(k) => k,
            Err(swapchain::SwapchainCreationError::ImageExtentNotSupported { .. }) => return Ok(()),
            Err(e) => {
                panic!("{e}");
            }
        };

        s.dirty = false;
        s.swapchain = swapchain;
        s.images = images;

        Ok(())
    }

    pub fn add_camera(
        &mut self,
        options: CameraOptions,
        width: u32,
        height: u32,
    ) -> anyhow::Result<Arc<Camera>> {
        let out_buffer = StorageImage::new(
            &self.buffer_allocator,
            vulkano::image::ImageDimensions::Dim2d {
                width,
                height,
                array_layers: 1,
            },
            vulkano::format::Format::R8G8B8A8_UNORM,
            Some(self.fallback_queue.queue_family_index()),
        )?;

        let camera = Arc::new(Camera {
            options: RefCell::new(options),
            out_buffer,
            idx: self.cameras.len() as u32,
            width,
            height,
        });

        self.cameras.push(camera.clone());

        Ok(camera)
    }

    pub fn draw_all(&self) -> anyhow::Result<()> {
        let mut futures = vec![];

        for camera in &self.cameras {
            let mut builder = AutoCommandBufferBuilder::primary(
                &self.command_allocator,
                self.fallback_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder.clear_color_image(ClearColorImageInfo {
                clear_value: ClearColorValue::Float([0.0, 0.0, 1.0, 1.0]),
                ..ClearColorImageInfo::image(camera.out_buffer.clone())
            })?;

            let command_buffer = builder.build()?;

            let future = sync::now(self.device.clone())
                .then_execute(self.fallback_queue.clone(), command_buffer)
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap();

            futures.push(future);
        }

        futures
            .into_iter()
            .for_each(|fence| fence.wait(None).unwrap());

        Ok(())
    }

    pub fn present_all(&mut self) {
        for swapchain_presenter in &mut self.swapchains {
            let SwapchainPresenter {
                swapchain,
                dirty,
                images,
                ..
            } = swapchain_presenter;

            if *dirty {
                continue;
            }

            let (image_i, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        *dirty = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

            if suboptimal {
                *dirty = true
            }

            let mut builder = AutoCommandBufferBuilder::primary(
                &self.command_allocator,
                self.fallback_queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            let camera = self.cameras[swapchain_presenter.camera_idx as usize].clone();
            builder
                .blit_image(BlitImageInfo::images(
                    camera.out_buffer.clone(),
                    images[image_i as usize].clone(),
                ))
                .unwrap();
            let command_buffer = builder.build().unwrap();

            let execution = sync::now(self.device.clone())
                .join(acquire_future)
                .then_execute(self.fallback_queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(
                    self.fallback_queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                )
                .then_signal_fence_and_flush();

            match execution {
                Ok(future) => {
                    future.wait(None).unwrap(); // wait for the GPU to finish
                }
                Err(FlushError::OutOfDate) => {
                    *dirty = true;
                }
                Err(e) => {
                    println!("Failed to flush future: {e}");
                }
            }
        }
    }

    pub fn save_png(&self, name: &str, camera_idx: u32) -> anyhow::Result<()> {
        let camera = self.cameras[camera_idx as usize].as_ref();

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
            (0..camera.width * camera.height * 4).map(|_| 0u8),
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_allocator,
            self.fallback_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder.copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            camera.out_buffer.clone(),
            buf.clone(),
        ))?;

        let command_buffer = builder.build()?;

        sync::now(self.device.clone())
            .then_execute(self.fallback_queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        use image::{ImageBuffer, Rgba};

        let buffer_content = buf.read().unwrap();
        let image =
            ImageBuffer::<Rgba<u8>, _>::from_raw(camera.width, camera.height, &buffer_content[..])
                .unwrap();
        image.save(name)?;

        Ok(())
    }
}

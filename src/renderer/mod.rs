use std::{cell::RefCell, sync::Arc};

use nalgebra_glm::{Quat, Vec3};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, ClearColorImageInfo,
        CommandBufferUsage, CopyImageToBufferInfo,
    },
    device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags},
    format::ClearColorValue,
    image::StorageImage,
    instance::Instance,
    memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    sync::{self, GpuFuture},
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

#[derive(Debug, Clone)]
pub struct Camera {
    pub idx: u32,
    pub width: u32,
    pub height: u32,

    pub options: RefCell<CameraOptions>,

    pub out_buffer: Arc<StorageImage>,
}

pub struct Renderer {
    pub cameras: Vec<Arc<Camera>>,

    // Allocators
    pub buffer_allocator: StandardMemoryAllocator,
    pub command_allocator: StandardCommandBufferAllocator,

    pub instance: Arc<Instance>,
    pub device: Arc<Device>,
    pub fallback_queue: Arc<Queue>,
}

impl Renderer {
    pub fn new(library: Arc<VulkanLibrary>) -> anyhow::Result<Self> {
        let instance = Instance::new(library, Default::default())?;
        let physical_device = instance.enumerate_physical_devices()?.next().unwrap();
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
            physical_device,
            DeviceCreateInfo {
                // here we pass the desired queue family to use by index
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;

        let queue = queues.next().unwrap();

        let buffer_allocator = StandardMemoryAllocator::new_default(device.clone());
        let command_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());

        Ok(Self {
            cameras: vec![],
            instance,
            device,
            fallback_queue: queue,
            buffer_allocator,
            command_allocator,
        })
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

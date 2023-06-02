mod renderer;

use std::sync::Arc;

pub use renderer::*;
use vulkano::VulkanLibrary;
use vulkano_win::create_surface_from_winit;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub fn main() -> anyhow::Result<()> {
    let library = VulkanLibrary::new()?;

    let mut renderer = Renderer::new(library)?;
    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new().build(&event_loop)?);
    let surface = create_surface_from_winit(window.clone(), renderer.instance.clone())?;

    let size = window.inner_size();

    let camera = renderer.add_camera(
        CameraOptions {
            fov: 120f32,
            near_plane: 0.1f32,
            far_plane: 4f32,
            ..Default::default()
        },
        32,
        size.width,
        size.height,
    )?;

    let swapchain_idx = renderer.attach_swapchain(camera.borrow().idx, surface);
    
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }

        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            renderer
                .refresh_swapchain(swapchain_idx, [size.width, size.height])
                .unwrap();
        }

        Event::RedrawRequested(_) => {
            renderer.draw_all().unwrap();
            renderer.present_all();
        }
        _ => (),
    });
}

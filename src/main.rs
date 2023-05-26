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
    let surface = create_surface_from_winit(window, renderer.instance.clone())?;

    let camera = renderer.add_camera(
        CameraOptions {
            fov: 120f32,
            ..Default::default()
        },
        800,
        600,
    )?;

    let renderer = Arc::new(renderer);
    renderer.draw_all()?;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::MainEventsCleared => {}
        _ => (),
    });
}

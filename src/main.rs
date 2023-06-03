mod renderer;
use std::sync::Arc;

use log::debug;
use nalgebra_glm::{quat_angle_axis, quat_euler_angles, quat_look_at_lh, Vec3};
pub use renderer::*;
use vulkano::VulkanLibrary;
use vulkano_win::create_surface_from_winit;
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub fn main() -> anyhow::Result<()> {
    drop(dotenv::dotenv());
    drop(color_eyre::install());
    drop(pretty_env_logger::init());

    let library = VulkanLibrary::new()?;

    let mut renderer = Renderer::new(library)?;
    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new().build(&event_loop)?);
    window.set_cursor_visible(false);
    drop(window.set_cursor_grab(winit::window::CursorGrabMode::Locked));

    let surface = create_surface_from_winit(window.clone(), renderer.instance.clone())?;

    let size = window.inner_size();

    let camera = renderer.add_camera(
        DynamicCameraData {
            fov: 60f32,
            near_plane: 0.1f32,
            far_plane: 4f32,
            pos: Vec3::zeros(),
            rotation: quat_look_at_lh(&Vec3::new(0., 0., 1.), &Vec3::new(0., 1., 0.)),
            samples: 4,
            ..Default::default()
        },
        2,
        size.width,
        size.height,
    )?;

    let swapchain_idx = renderer.attach_swapchain(camera.borrow().idx, surface);
    let mouse_speed = 0.001;

    let mut dt = 0.;
    let mut last_frame = std::time::Instant::now();
    let mut fps_timer = 0.;
    let mut fps_counter = 0;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }

        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion { delta },
            ..
        } => {
            let (x, y) = delta;
            let mut cam = camera.borrow_mut();
            let mut opts = cam.dynamic_data.borrow_mut();

            opts.rotation *= quat_angle_axis(x as f32 * mouse_speed, &Vec3::new(0., 1., 0.));
            drop(opts);

            cam.refresh_data_buffer().unwrap();
        }

        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            renderer
                .refresh_swapchain(swapchain_idx, [size.width, size.height])
                .unwrap();
        }

        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::K),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                },
            ..
        } => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            let filename = format!("{}-screenshot.png", now.as_secs()).to_string();
            renderer.save_png(&filename, camera.borrow().idx).unwrap();
        }

        Event::MainEventsCleared => {
            let now = std::time::Instant::now();
            dt = (now - last_frame).as_secs_f32();
            last_frame = now;
            fps_timer += dt;
            
            if fps_timer > 1.0 {
                debug!("FPS: {}", fps_counter);
                fps_counter = 0;
                fps_timer = 0.;
            }

            renderer.draw_all().unwrap();
            renderer.present_all().unwrap();
            fps_counter += 1;
        }
        _ => (),
    });
}

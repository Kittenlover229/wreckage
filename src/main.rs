mod renderer;
use std::sync::Arc;

use egui_winit_vulkano::{
    egui::{epaint::Shadow, FontDefinitions, RawInput, Visuals},
    Gui, GuiConfig,
};
use log::debug;
use nalgebra_glm::{pi, quat_angle_axis, quat_look_at_lh, Vec3};
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
    let instance = renderer.instance.clone();
    let window = Arc::new(WindowBuilder::new().build(&event_loop)?);
    let surface = create_surface_from_winit(window.clone(), instance)?;

    let mut gui = Gui::new(
        &event_loop,
        surface.clone(),
        renderer.fallback_queue.clone(),
        GuiConfig {
            is_overlay: true,
            preferred_format: Some(
                renderer
                    .physical
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            ),
            ..Default::default()
        },
    );

    let egui_visuals = Visuals {
        window_shadow: Shadow::NONE,
        ..Default::default()
    };
    gui.context().set_visuals(egui_visuals);

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
        4,
        size.width,
        size.height,
    )?;

    let swapchain_idx = renderer.attach_swapchain(camera.borrow().idx, surface);
    let mouse_speed = 0.001;

    let mut dt = 0.;
    let mut last_frame = std::time::Instant::now();
    let mut fps_timer = 0.;
    let mut fps_counter = 0;
    let mut x_rot_accum = 0f32;
    let mut y_rot_accum = 0f32;

    event_loop.run(move |event, _, control_flow| {
        match &event {
            Event::WindowEvent { event, .. } => {
                if gui.update(&event) {
                    return;
                }
            }
            _ => {}
        }

        match event {
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
                let x = x as f32;
                let y = y as f32;

                let mut cam = camera.borrow_mut();
                let mut opts = cam.dynamic_data.borrow_mut();

                x_rot_accum += x * mouse_speed;
                y_rot_accum += y * mouse_speed;
                let half_pi = pi::<f32>() / 2.;

                if y_rot_accum > half_pi {
                    y_rot_accum = half_pi;
                } else if y_rot_accum < -half_pi {
                    y_rot_accum = -half_pi;
                }

                opts.rotation = quat_angle_axis(y_rot_accum, &Vec3::new(1., 0., 0.))
                    * quat_angle_axis(x_rot_accum, &Vec3::new(0., 1., 0.));
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

                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    egui_winit_vulkano::egui::Window::new("Hello, world!").show(&ctx, |_| ());
                    // Fill egui UI layout here
                });

                let draw_future = renderer.draw_all().unwrap();
                renderer.present(0, draw_future, &mut gui).unwrap();
                fps_counter += 1;
            }
            _ => (),
        }
    });
}

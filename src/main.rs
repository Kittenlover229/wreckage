mod renderer;
use std::{collections::VecDeque, sync::Arc};

use egui_winit_vulkano::{
    egui::{epaint::Shadow, Visuals},
    Gui, GuiConfig,
};
use nalgebra_glm::{pi, quat_angle_axis, quat_look_at_lh, Vec3};
pub use renderer::*;
use vulkano::VulkanLibrary;
use vulkano_win::create_surface_from_winit;
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub struct Benchmark {
    capacity: usize,
    data: VecDeque<f64>,
    fps: u32,
}

use egui_winit_vulkano::egui::plot::*;
use egui_winit_vulkano::egui::*;

impl Benchmark {
    pub fn new(capacity: usize) -> Self {
        Self {
            // TODO: use fps cap as capacity
            capacity,
            fps: 0,
            data: VecDeque::with_capacity(capacity),
        }
    }

    pub fn draw(&self, ui: &mut Ui) {
        let iter = self
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| [i as f64, *v * 1000.0]);
        let curve = Line::new(PlotPoints::from_iter(iter)).color(Color32::BLUE);
        let ok = HLine::new(1000.0 / 30.0).color(Color32::GREEN);
        let bad = HLine::new(1000.0 / 60.0).color(Color32::RED);

        ui.label("Frametime (Draw + Present)");
        ui.label(format!("FPS: {}", self.fps));
        Plot::new("plot")
            .view_aspect(2.0)
            .include_y(0)
            .show(ui, |plot_ui| {
                plot_ui.line(curve);
                plot_ui.hline(ok);
                plot_ui.hline(bad);
            });
        ui.label("Green line is frametime for 30 FPS");
        ui.label("Red line is frametime for 60 FPS");
    }

    pub fn push(&mut self, v: f64) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(v);
    }
}

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

    let mut egui_bench = Benchmark::new(250);

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
            samples: 2,
            ..Default::default()
        },
        8,
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
    let mut freeze = false;

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
                if freeze {
                    return;
                }

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
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    },
                ..
            } => {
                freeze = !freeze;
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
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    egui_winit_vulkano::egui::Window::new("Benchmark")
                        .default_height(600.0)
                        .show(&ctx, |ui| {
                            egui_bench.draw(ui);
                        });
                });

                let now = std::time::Instant::now();
                dt = (now - last_frame).as_secs_f32();
                last_frame = now;

                let draw_future = renderer.draw_all().unwrap();
                renderer.present(0, draw_future, &mut gui).unwrap();

                if !freeze {
                    egui_bench.push(last_frame.elapsed().as_secs_f64());
                }

                fps_timer += dt;
                if fps_timer > 1.0 {
                    egui_bench.fps = fps_counter;
                    fps_counter = 0;
                    fps_timer = 0.;
                }
                fps_counter += 1;
            }
            _ => (),
        }
    });
}

mod renderer;

use std::sync::Arc;

use nalgebra_glm::{quat_angle_axis, quat_look_at_lh, quat_rotation, rotate_normalized_axis, Vec3, quat_euler_angles};
pub use renderer::*;
use vulkano::VulkanLibrary;
use vulkano_win::create_surface_from_winit;
use winit::{
    event::{DeviceEvent, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub fn main() -> anyhow::Result<()> {
    let library = VulkanLibrary::new()?;

    let mut renderer = Renderer::new(library)?;
    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new().build(&event_loop)?);
    window.set_cursor_visible(false);
    drop(window.set_cursor_grab(winit::window::CursorGrabMode::Locked));

    let surface = create_surface_from_winit(window.clone(), renderer.instance.clone())?;

    let size = window.inner_size();

    let camera = renderer.add_camera(
        CameraOptions {
            fov: 60f32,
            near_plane: 0.1f32,
            far_plane: 4f32,
            pos: Vec3::zeros(),
            // The order of arguments is reversed because by default the camera
            // is looking upwards, by swapping the UP and FORWARD directions we
            // actually make it look forward
            rotation: quat_look_at_lh(&Vec3::new(0., 0., 1.), &Vec3::new(0., 1., 0.)),
            ..Default::default()
        },
        1,
        size.width,
        size.height,
    )?;

    let swapchain_idx = renderer.attach_swapchain(camera.borrow().idx, surface);
    let mouse_speed = 0.001;

    let mut dt = 0.;
    let mut last_frame = std::time::Instant::now();

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
            let mut opts = cam.options.borrow_mut();

            opts.rotation *= quat_angle_axis(x as f32 * mouse_speed, &Vec3::new(0., 1., 0.));
            //opts.rotation *= quat_angle_axis(-y as f32 * mouse_speed, &Vec3::new(1., 0., 0.));
            println!("{}", quat_euler_angles(&opts.rotation));

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

        Event::MainEventsCleared => {
            let now = std::time::Instant::now();
            dt = (now - last_frame).as_secs_f32();
            last_frame = now;

            renderer.draw_all().unwrap();
            renderer.present_all().unwrap();
        }
        _ => (),
    });
}

mod renderer;

pub use renderer::*;
use vulkano::VulkanLibrary;

pub fn main() -> anyhow::Result<()> {
    let library = VulkanLibrary::new()?;

    let mut renderer = Renderer::new(library)?;

    let _camera = renderer.add_camera(Default::default(), Default::default(), 120f32, 800, 600)?;
    renderer.draw_all()?;
    renderer.save_png("hi.png", _camera.idx)?;

    println!("Hello, world!");

    Ok(())
}

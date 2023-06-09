pub mod main {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/renderer/shaders/main.comp",
    }
}

pub mod sort {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "./src/renderer/shaders/sort.comp",
    }
}
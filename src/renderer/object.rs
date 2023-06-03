use nalgebra_glm::{Quat, Vec3};
use vulkano::buffer::BufferContents;

#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: Vec3,
}

impl Material {
    pub const ERROR_PINK: Material = Material {
        albedo: Vec3::new(
            // Deep pink, nice colour
            1., 0.078, 0.576,
        ),
    };
}

pub const MATERIAL_TABLE_MAX_SIZE: usize = 4;

#[derive(BufferContents)]
#[repr(C)]
pub struct MaterialTableBuffer {
    pub albedo: [[f32; 3]; MATERIAL_TABLE_MAX_SIZE],
}

impl<'a> FromIterator<&'a Material> for MaterialTableBuffer {
    fn from_iter<T: IntoIterator<Item = &'a Material>>(iter: T) -> Self {
        let mut albedo = [[0f32; 3]; MATERIAL_TABLE_MAX_SIZE];

        for (i, mat) in iter.into_iter().take(MATERIAL_TABLE_MAX_SIZE).enumerate() {
            albedo[i] = mat.albedo.data.0[0];
        }

        Self { albedo }
    }
}

impl Default for MaterialTableBuffer {
    fn default() -> Self {
        Self::from_iter(&[Material::ERROR_PINK])
    }
}

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

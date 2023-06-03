use nalgebra_glm::{Quat, Vec3};

#[derive(Debug, Clone)]
pub struct Material {
    
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

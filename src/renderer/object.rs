use nalgebra_glm::{vec3, Quat, Vec3};
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

// TODO: Maybe rename this?
#[derive(Debug, Clone, BufferContents, Default, Copy)]
#[repr(C)]
pub struct BVHAABB {
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub aabb_center: [f32; 3],
    pub morton: u32,

    pub object_id: u32,
    pub left_idx: u32,
    pub right_idx: u32,
}

#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub struct BoundingVolumeHierarchy {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub len: u32,

    pub volumes: [BVHAABB; 4096],
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

impl RenderableObject {
    pub fn into_buffer(self, object_id: u32) -> (SphereBufferData, BVHAABB) {
        let sphere = SphereBufferData {
            center: self.pos.data.0[0],
            radius: match self.kind {
                RenderableObjectKind::Sphere { radius } => radius,
                _ => unimplemented!(),
            },
        };

        let radius = sphere.radius;

        let bvh = BVHAABB {
            aabb_min: (self.pos - vec3(radius, radius, radius) / 2.).data.0[0],
            aabb_max: (self.pos + vec3(radius, radius, radius) / 2.).data.0[0],
            aabb_center: sphere.center,
            morton: 0b0,
            object_id,
            left_idx: 0,
            right_idx: 0,
        };
        (sphere, bvh)
    }
}

// TESTING GARBAGE
pub fn generate_object_grid(side: usize) -> (Vec3, Vec<(SphereBufferData, BVHAABB)>, Vec3) {
    let mut out = vec![];

    for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                let obj = RenderableObject {
                    kind: RenderableObjectKind::Sphere {
                        radius: ((i * j ^ k) % 5 + 1) as f32 / 10.,
                    },
                    pos: vec3(i as f32, j as f32, k as f32),
                    rotation: Quat::identity(),
                };

                out.push(obj.into_buffer((i * side * side + j * side + k) as u32));
            }
        }
    }

    (
        -Vec3::new(1., 1., 1.),
        out,
        Vec3::new(1., 1., 1.) * (side as f32 + 1f32),
    )
}

#[derive(Debug, Clone, BufferContents, Default, Copy)]
#[repr(C)]
pub struct SphereBufferData {
    pub center: [f32; 3],
    pub radius: f32,
}

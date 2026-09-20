use tangle_core::{
    FiberAnchor, FiberAssembly, FiberId, JunctionId, JunctionParameterId, PeriodicCell, Section,
};

fn main() {
    let mut assembly = FiberAssembly::new(PeriodicCell::orthorhombic([1.0, 1.0, 1.0], [false; 3]));
    let material = assembly.materials.add("carbon fiber");
    let section = assembly.sections.add(Section::Circular { radius: 5.0e-6 });
    let centerlines = [
        (
            FiberId(10),
            [[-4.0e-4, 0.0, 0.0], [4.0e-4, 0.0, 0.0]],
            [[0.4996, 0.5, 0.5], [0.5004, 0.5, 0.5]],
        ),
        (
            FiberId(20),
            [[0.0, -4.0e-4, 0.0], [0.0, 4.0e-4, 0.0]],
            [[0.5, 0.4996, 0.5], [0.5, 0.5004, 0.5]],
        ),
    ];
    for (id, intrinsic, placed) in centerlines {
        assembly
            .add_fiber(id, material, section, &intrinsic, &placed)
            .unwrap();
    }

    let law = assembly.junction_laws.add("welded");
    let anchors = [FiberId(10), FiberId(20)].map(|fiber| FiberAnchor {
        fiber,
        rest_arc_length: 4.0e-4,
        section_offset: None,
    });
    assembly
        .add_junction(JunctionId(1), law, JunctionParameterId(0), &anchors)
        .unwrap();
    assembly.validate().unwrap();

    println!(
        "assembly: {} fibers, {} persistent junction",
        assembly.topology.fibers.len(),
        assembly.junctions.junctions.len()
    );
    for anchor in anchors {
        println!(
            "  {anchor:?} -> {:?}",
            assembly.resolve_anchor(anchor).unwrap()
        );
    }
}

use super::*;

/// Supply settlement clearings without coupling the vegetation crate to buildings.
pub(super) fn stream_vegetation(
    mut field: ResMut<vegetation::VegetationField>,
    rivers: Res<rivers::RiverField>,
    buildings: Res<buildings::BuildingField>,
    players: Query<&Player>,
    time: Res<Time>,
) {
    let planner = buildings.planner().cloned();
    let clearings_around = move |network: Option<
        &std::sync::Arc<rivers::network::RiverNetwork>,
    >,
                                 chunk| {
        match &planner {
            Some(planner) => buildings::clearings_around(planner, network, chunk),
            None => Vec::new(),
        }
    };
    let player = players.single().ok();
    vegetation::streamVegetation(
        &mut field,
        &rivers,
        player,
        time.delta_secs(),
        clearings_around,
    );
}

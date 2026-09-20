/// Information required to reproduce or audit an assembly.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Provenance {
    /// Generator or importer name.
    pub source: String,
    /// Version of the generation algorithm or import schema.
    pub version: String,
    /// Deterministic random seed, when generation used randomness.
    pub seed: Option<u64>,
    /// Human-readable notes about process stages or assumptions.
    pub notes: Vec<String>,
}

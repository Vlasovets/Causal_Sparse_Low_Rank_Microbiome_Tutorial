# Shared phylum colour palette used by all NetCoMi network plots.
# Maps both SILVA 128 and SILVA 138 phylum names to the same colour so
# networks from different datasets can be compared directly.

PHYLUM_PALETTE <- c(
  # Core gut phyla — SILVA 128 names / SILVA 138 names both covered
  "Firmicutes"          = "#117733",   # dark green
  "Bacteroidetes"       = "#CC6677",   # rose     (SILVA 128)
  "Bacteroidota"        = "#CC6677",   # rose     (SILVA 138)
  "Actinobacteria"      = "#88CCEE",   # light blue (SILVA 128)
  "Actinobacteriota"    = "#88CCEE",   # light blue (SILVA 138)
  "Proteobacteria"      = "#AA4499",   # mauve
  "Verrucomicrobia"     = "#999933",   # olive    (SILVA 128)
  "Verrucomicrobiota"   = "#999933",   # olive    (SILVA 138)
  "Tenericutes"         = "#44AA99",   # teal     (SILVA 128, reclassified in 138)
  "Fusobacteria"        = "#332288",   # dark purple (SILVA 128)
  "Fusobacteriota"      = "#332288",   # dark purple (SILVA 138)
  "Cyanobacteria"       = "#BBCC77",   # yellow-green
  "Desulfobacterota"    = "#882255",   # wine
  "Desulfobacterota"    = "#882255",
  "Euryarchaeota"       = "#661100",   # dark red
  "Unknown"             = "#AAAAAA"    # grey fallback
)

# Returns the colour for a given phylum name (works with both SILVA versions)
phylum_colour <- function(phylum_name) {
  col <- PHYLUM_PALETTE[phylum_name]
  if (is.na(col)) "#AAAAAA" else col
}

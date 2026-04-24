# hole_base_layer.R
# Provides `hole_base_layer()`: a ggplot2 object with the Par-4 hole layout
# as the background, ready for you to add your own layers on top.
#
# Usage:
#   source("r_hole_layout/hole_base_layer.R")
#
#   hole_base_layer() +
#     geom_point(data = my_data, aes(x = x, y = y, colour = esho_mean))
#
# Optional arguments:
#   layout_dir  – path to the folder containing the CSVs (default: same dir as this file)
#   show_grid   – overlay the strategy grid points (default FALSE)
#   show_pin    – mark the hole pin (default TRUE)
#   show_tee    – mark the tee point (default TRUE)
#   alpha       – polygon fill transparency (default 0.6)

library(ggplot2)
library(dplyr)
library(readr)

# ---------------------------------------------------------------------------
# Colour + draw-order config (mirrors Python _LIE_COLORS)
# ---------------------------------------------------------------------------
.LIE_COLOURS <- c(
  "OB"           = "lightcoral",
  "rough"        = "mediumseagreen",
  "water_hazard" = "skyblue",
  "new_hazard3"  = "skyblue",
  "fairway"      = "forestgreen",
  "new_fairway"  = "forestgreen",
  "bunker"       = "tan",
  "green"        = "lightgreen",
  "tee"          = "darkgreen"
)

# Drawn back-to-front so foreground lies sit on top
.LIE_ORDER <- c("OB", "rough", "water_hazard", "new_hazard3",
                 "fairway", "new_fairway", "bunker", "green", "tee")

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
hole_base_layer <- function(
    layout_dir = dirname(sys.frame(1)$ofile),
    show_grid  = FALSE,
    show_pin   = TRUE,
    show_tee   = TRUE,
    alpha      = 0.6
) {
  polys <- read_csv(file.path(layout_dir, "hole_polygons.csv"), show_col_types = FALSE) |>
    mutate(lie = factor(lie, levels = .LIE_ORDER)) |>
    arrange(lie)

  pts <- read_csv(file.path(layout_dir, "hole_points.csv"), show_col_types = FALSE)

  p <- ggplot() +
    geom_polygon(
      data      = polys,
      aes(x = x, y = y, group = polygon_id, fill = lie),
      colour    = "black",
      linewidth = 0.25,
      alpha     = alpha
    ) +
    scale_fill_manual(
      values   = .LIE_COLOURS,
      name     = "Lie",
      na.value = "lightgrey",
      # Collapse duplicate labels (new_fairway / new_hazard3 share colours)
      labels   = c(
        "OB"           = "OB",
        "rough"        = "Rough",
        "water_hazard" = "Water",
        "new_hazard3"  = "Water",
        "fairway"      = "Fairway",
        "new_fairway"  = "Fairway",
        "bunker"       = "Bunker",
        "green"        = "Green",
        "tee"          = "Tee"
      )
    ) +
    coord_equal() +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      legend.position  = "right"
    ) +
    labs(x = "x (yards, left/right of centreline)", y = "y (yards from tee)")

  if (show_tee) {
    tee <- filter(pts, label == "Tee")
    p <- p + geom_point(
      data   = tee,
      aes(x = x, y = y),
      shape  = 4, size = 4, colour = "red", stroke = 1.5
    ) +
      annotate("text", x = tee$x + 3, y = tee$y - 4,
               label = "Tee", size = 3, colour = "red")
  }

  if (show_pin) {
    pin <- filter(pts, label == "Pin")
    p <- p + geom_point(
      data  = pin,
      aes(x = x, y = y),
      shape = 21, size = 4, fill = "white", colour = "black", stroke = 1.5
    ) +
      annotate("text", x = pin$x + 3, y = pin$y + 3,
               label = "Pin", size = 3, colour = "black")
  }

  if (show_grid) {
    grid <- read_csv(file.path(layout_dir, "strategy_grid.csv"), show_col_types = FALSE)
    p <- p + geom_point(
      data  = grid,
      aes(x = x, y = y),
      size  = 0.8, alpha = 0.35, colour = "grey20"
    )
  }

  p
}

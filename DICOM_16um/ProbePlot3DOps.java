import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.Arrays;

public class ProbePlot3DOps {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics.mph";

  public static void main(String[] args) throws Exception {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MPH, e);
    }

    String pg = "pg3d_probe";
    try {
      m.result().remove(pg);
    } catch (Exception ignored) {
    }
    m.result().create(pg, "PlotGroup3D");
    try {
      m.result(pg).set("data", "dset4");
    } catch (Exception ignored) {
    }

    String[] candidates = new String[]{
        "Surface",
        "Point",
        "PointGraph",
        "PointVolume",
        "Volume",
        "Slice",
        "Multislice",
        "Contour",
        "Mesh",
        "ArrowVolume",
        "ArrowSurface",
        "ArrowLine",
        "LineGraph",
        "Isosurface",
        "Image",
        "Streamline"
    };

    for (String op : candidates) {
      try {
        try {
          m.result(pg).feature().remove("f1");
        } catch (Exception ignored) {
        }
        m.result(pg).create("f1", op);
        System.out.println("PG3_OK op=" + op + " type=" + m.result(pg).feature("f1").getType());
        try {
          String[] exprAllowed = m.result(pg).feature("f1").getAllowedPropertyValues("expr");
          if (exprAllowed != null) {
            System.out.println("  expr allowed count=" + exprAllowed.length);
          }
        } catch (Exception ignored) {
        }
      } catch (Exception e) {
        System.out.println("PG3_BAD op=" + op + " msg=" + e.getMessage());
      }
    }

    try {
      m.result().remove(pg);
    } catch (Exception ignored) {
    }
    System.out.println("datasets=" + Arrays.toString(m.result().dataset().tags()));
  }
}

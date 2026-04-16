import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class FixLivytanVisibility {
  private static final String FILE =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";

  private static void p(String s) {
    System.out.println(s);
  }

  private static boolean hasResult(Model m, String tag) {
    try {
      m.result(tag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean hasResultFeature(Model m, String pg, String ft) {
    try {
      m.result(pg).feature(ft);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static String pickDataset(Model m) {
    try {
      String[] ds = m.result().dataset().tags();
      for (String d : ds) {
        if ("dset1".equals(d)) return d;
      }
      if (ds.length > 0) return ds[0];
    } catch (Exception ignored) {}
    return null;
  }

  public static void main(String[] args) throws Exception {
    Model m;
    try {
      m = ModelUtil.load("Model", FILE);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load " + FILE, e);
    }

    String dset = pickDataset(m);
    p("VIS_DATASET|" + dset);

    // A plain visible surface plot so geometry is immediately visible in Results.
    if (!hasResult(m, "pg_geom_preview")) m.result().create("pg_geom_preview", "PlotGroup3D");
    m.result("pg_geom_preview").label("Livytan Geometry Preview");
    if (dset != null) {
      try { m.result("pg_geom_preview").set("data", dset); } catch (Exception ignored) {}
    }
    if (!hasResultFeature(m, "pg_geom_preview", "surf1")) m.result("pg_geom_preview").create("surf1", "Surface");
    try { m.result("pg_geom_preview").feature("surf1").set("expr", "1"); } catch (Exception ignored) {}
    try { m.result("pg_geom_preview").feature("surf1").set("descr", "Geometry visibility surface"); } catch (Exception ignored) {}
    try { m.result("pg_geom_preview").feature("surf1").selection().all(); } catch (Exception ignored) {}
    try { m.result("pg_geom_preview").run(); } catch (Exception ignored) {}

    // Keep a stress plot group as well.
    if (!hasResult(m, "pg_vms_preview")) m.result().create("pg_vms_preview", "PlotGroup3D");
    m.result("pg_vms_preview").label("Livytan Von Mises Preview");
    if (dset != null) {
      try { m.result("pg_vms_preview").set("data", dset); } catch (Exception ignored) {}
    }
    if (!hasResultFeature(m, "pg_vms_preview", "surf1")) m.result("pg_vms_preview").create("surf1", "Surface");
    try { m.result("pg_vms_preview").feature("surf1").set("expr", "solid.mises"); } catch (Exception ignored) {}
    try { m.result("pg_vms_preview").feature("surf1").set("unit", "Pa"); } catch (Exception ignored) {}
    try { m.result("pg_vms_preview").feature("surf1").selection().all(); } catch (Exception ignored) {}
    try { m.result("pg_vms_preview").run(); } catch (Exception ignored) {}

    try {
      m.save(FILE);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save " + FILE, e);
    }
    p("SAVED|" + FILE);
  }
}

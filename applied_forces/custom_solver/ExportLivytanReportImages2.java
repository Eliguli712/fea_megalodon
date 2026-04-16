import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ExportLivytanReportImages2 {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";
  private static final String IMG_GEOM =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_pg_geom_preview.png";
  private static final String IMG_VMS =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_pg_vms_preview.png";

  private static void p(String s) { System.out.println(s); }

  private static boolean hasResult(Model m, String tag) {
    try { m.result(tag); return true; } catch (Exception e) { return false; }
  }

  private static boolean hasResultFeature(Model m, String pg, String ft) {
    try { m.result(pg).feature(ft); return true; } catch (Exception e) { return false; }
  }

  private static String pickDataset(Model m, String preferred) {
    try {
      String[] ds = m.result().dataset().tags();
      for (String d : ds) if (preferred.equals(d)) return d;
      if (ds.length > 0) return ds[0];
    } catch (Exception ignored) {}
    return null;
  }

  private static void exportImage(Model m, String tag, String pg, String out) {
    try { m.result().export().remove(tag); } catch (Exception ignored) {}
    m.result().export().create(tag, "Image");
    m.result().export(tag).set("plotgroup", pg);
    try { m.result().export(tag).set("imagetype", "png"); } catch (Exception ignored) {}
    try { m.result().export(tag).set("size", "manual"); } catch (Exception ignored) {}
    try { m.result().export(tag).set("unit", "px"); } catch (Exception ignored) {}
    try { m.result().export(tag).set("width", 1200); } catch (Exception ignored) {}
    try { m.result().export(tag).set("height", 900); } catch (Exception ignored) {}
    try { m.result().export(tag).set("antialias", "off"); } catch (Exception ignored) {}
    m.result().export(tag).set("pngfilename", out);
    m.result().export(tag).run();
  }

  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", MPH); }
    catch (IOException e) { throw new RuntimeException("load failed", e); }

    String dsetGeom = pickDataset(m, "dset1");
    String dsetVms = pickDataset(m, "dset2");
    if (dsetVms == null) dsetVms = dsetGeom;

    if (!hasResult(m, "pg_geom_preview")) m.result().create("pg_geom_preview", "PlotGroup3D");
    if (dsetGeom != null) { try { m.result("pg_geom_preview").set("data", dsetGeom); } catch (Exception ignored) {} }
    if (!hasResultFeature(m, "pg_geom_preview", "surf1")) m.result("pg_geom_preview").create("surf1", "Surface");
    try { m.result("pg_geom_preview").feature("surf1").set("expr", "1"); } catch (Exception ignored) {}
    try { m.result("pg_geom_preview").feature("surf1").selection().all(); } catch (Exception ignored) {}
    try { m.result("pg_geom_preview").run(); } catch (Exception ignored) {}

    if (!hasResult(m, "pg_vms_preview")) m.result().create("pg_vms_preview", "PlotGroup3D");
    if (dsetVms != null) { try { m.result("pg_vms_preview").set("data", dsetVms); } catch (Exception ignored) {} }
    if (!hasResultFeature(m, "pg_vms_preview", "surf1")) m.result("pg_vms_preview").create("surf1", "Surface");
    try { m.result("pg_vms_preview").feature("surf1").set("expr", "solid.mises"); } catch (Exception ignored) {}
    try { m.result("pg_vms_preview").feature("surf1").set("unit", "Pa"); } catch (Exception ignored) {}
    try { m.result("pg_vms_preview").feature("surf1").selection().all(); } catch (Exception ignored) {}
    try { m.result("pg_vms_preview").run(); } catch (Exception ignored) {}

    exportImage(m, "img_geom_preview", "pg_geom_preview", IMG_GEOM);
    exportImage(m, "img_vms_preview", "pg_vms_preview", IMG_VMS);

    p("EXPORTED|" + IMG_GEOM);
    p("EXPORTED|" + IMG_VMS);

    try { m.save(MPH); } catch (IOException e) { throw new RuntimeException("save failed", e); }
    p("SAVED|" + MPH);
  }
}

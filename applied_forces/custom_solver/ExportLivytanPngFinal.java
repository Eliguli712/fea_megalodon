import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;

public class ExportLivytanPngFinal {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";
  private static final String IMG_GEOM =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_pg_geom_preview_comsol.png";
  private static final String IMG_VMS =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_pg_vms_preview_comsol.png";

  private static void p(String s) { System.out.println(s); }

  private static boolean hasResult(Model m, String tag) {
    try { m.result(tag); return true; } catch (Exception e) { return false; }
  }

  private static boolean hasFeature(Model m, String pg, String ft) {
    try { m.result(pg).feature(ft); return true; } catch (Exception e) { return false; }
  }

  private static String pickDataset(Model m, String preferred) {
    try {
      String[] d = m.result().dataset().tags();
      for (String t : d) if (preferred.equals(t)) return t;
      if (d.length > 0) return d[0];
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
    try { m.result().export(tag).set("width", 900); } catch (Exception ignored) {}
    try { m.result().export(tag).set("height", 700); } catch (Exception ignored) {}
    try { m.result().export(tag).set("antialias", "off"); } catch (Exception ignored) {}
    m.result().export(tag).set("pngfilename", out);

    p("EXPORT_BEGIN|" + tag + "|" + LocalDateTime.now());
    m.result().export(tag).run();
    p("EXPORT_DONE|" + tag + "|" + LocalDateTime.now());

    File f = new File(out);
    p("FILE_SIZE|" + out + "|" + f.length());
  }

  public static void main(String[] args) {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model", e);
    }

    String dsetGeom = pickDataset(m, "dset1");
    String dsetVms = pickDataset(m, "dset2");
    if (dsetVms == null) dsetVms = dsetGeom;
    p("DATASET_GEOM|" + dsetGeom);
    p("DATASET_VMS|" + dsetVms);

    if (!hasResult(m, "pg_geom_preview")) m.result().create("pg_geom_preview", "PlotGroup3D");
    try { m.result("pg_geom_preview").label("Livytan Geometry Preview"); } catch (Exception ignored) {}
    if (dsetGeom != null) {
      try { m.result("pg_geom_preview").set("data", dsetGeom); } catch (Exception ignored) {}
    }
    if (!hasFeature(m, "pg_geom_preview", "surf1")) m.result("pg_geom_preview").create("surf1", "Surface");
    try { m.result("pg_geom_preview").feature("surf1").set("expr", "1"); } catch (Exception ignored) {}
    try { m.result("pg_geom_preview").feature("surf1").selection().all(); } catch (Exception ignored) {}
    try { m.result("pg_geom_preview").run(); } catch (Exception ignored) {}

    if (!hasResult(m, "pg_vms_preview")) m.result().create("pg_vms_preview", "PlotGroup3D");
    try { m.result("pg_vms_preview").label("Livytan Von Mises Preview"); } catch (Exception ignored) {}
    if (dsetVms != null) {
      try { m.result("pg_vms_preview").set("data", dsetVms); } catch (Exception ignored) {}
    }
    if (!hasFeature(m, "pg_vms_preview", "surf1")) m.result("pg_vms_preview").create("surf1", "Surface");
    try { m.result("pg_vms_preview").feature("surf1").set("expr", "solid.mises"); } catch (Exception ignored) {}
    try { m.result("pg_vms_preview").feature("surf1").set("unit", "Pa"); } catch (Exception ignored) {}
    try { m.result("pg_vms_preview").feature("surf1").selection().all(); } catch (Exception ignored) {}
    try { m.result("pg_vms_preview").run(); } catch (Exception ignored) {}

    exportImage(m, "img_geom_final", "pg_geom_preview", IMG_GEOM);
    exportImage(m, "img_vms_final", "pg_vms_preview", IMG_VMS);

    try {
      m.save(MPH);
      p("SAVED|" + MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save model", e);
    }
  }
}

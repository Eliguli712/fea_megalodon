import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ExportLivytanReportImages {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";
  private static final String IMG_GEOM =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_pg_geom_preview.png";
  private static final String IMG_VMS =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_pg_vms_preview.png";

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

  private static String pickDataset(Model m, String preferred) {
    try {
      String[] dsets = m.result().dataset().tags();
      for (String d : dsets) {
        if (preferred.equals(d)) return d;
      }
      if (dsets.length > 0) return dsets[0];
    } catch (Exception ignored) {}
    return null;
  }

  private static void exportImage(Model m, String exportTag, String plotGroup, String filePath) {
    try { m.result().export().remove(exportTag); } catch (Exception ignored) {}
    m.result().export().create(exportTag, "Image3D");
    m.result().export(exportTag).set("plotgroup", plotGroup);
    try { m.result().export(exportTag).set("imagetype", "png"); } catch (Exception ignored) {}
    try { m.result().export(exportTag).set("qualitylevel", "95"); } catch (Exception ignored) {}
    try { m.result().export(exportTag).set("unit", "px"); } catch (Exception ignored) {}
    try { m.result().export(exportTag).set("width", 1400); } catch (Exception ignored) {}
    try { m.result().export(exportTag).set("height", 980); } catch (Exception ignored) {}
    m.result().export(exportTag).set("pngfilename", filePath);
    m.result().export(exportTag).run();
  }

  public static void main(String[] args) throws Exception {
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
    m.result("pg_geom_preview").label("Livytan Geometry Preview");
    if (dsetGeom != null) {
      try { m.result("pg_geom_preview").set("data", dsetGeom); } catch (Exception ignored) {}
    }
    if (!hasResultFeature(m, "pg_geom_preview", "surf1")) m.result("pg_geom_preview").create("surf1", "Surface");
    try { m.result("pg_geom_preview").feature("surf1").set("expr", "1"); } catch (Exception ignored) {}
    try { m.result("pg_geom_preview").feature("surf1").selection().all(); } catch (Exception ignored) {}
    m.result("pg_geom_preview").run();

    if (!hasResult(m, "pg_vms_preview")) m.result().create("pg_vms_preview", "PlotGroup3D");
    m.result("pg_vms_preview").label("Livytan Von Mises Preview");
    if (dsetVms != null) {
      try { m.result("pg_vms_preview").set("data", dsetVms); } catch (Exception ignored) {}
    }
    if (!hasResultFeature(m, "pg_vms_preview", "surf1")) m.result("pg_vms_preview").create("surf1", "Surface");
    try { m.result("pg_vms_preview").feature("surf1").set("expr", "solid.mises"); } catch (Exception ignored) {}
    try { m.result("pg_vms_preview").feature("surf1").set("unit", "Pa"); } catch (Exception ignored) {}
    try { m.result("pg_vms_preview").feature("surf1").selection().all(); } catch (Exception ignored) {}
    m.result("pg_vms_preview").run();

    exportImage(m, "img_geom_preview", "pg_geom_preview", IMG_GEOM);
    exportImage(m, "img_vms_preview", "pg_vms_preview", IMG_VMS);

    p("EXPORTED|" + IMG_GEOM);
    p("EXPORTED|" + IMG_VMS);

    try {
      m.save(MPH);
      p("SAVED|" + MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save model", e);
    }
  }
}

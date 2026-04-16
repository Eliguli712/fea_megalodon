import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

public class ExportContrastSeparatedViews {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
  private static final String OUT_DIR = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/exports/holocastic_full_body_images";
  private static final String COLOR_VM = "Turbo";
  private static final String COLOR_MR = "Prism";

  private static double evalMax(Model m, String tag, String dset) {
    try { m.result().numerical().remove(tag); } catch (Exception ignored) {}
    try {
      m.result().numerical().create(tag, "MaxVolume");
      m.result().numerical(tag).set("expr", new String[]{"solid.mises"});
      m.result().numerical(tag).set("unit", new String[]{"Pa"});
      m.result().numerical(tag).set("data", dset);
      m.result().numerical(tag).selection().all();
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) return r[0][0];
    } catch (Exception e) {
      System.out.println("MAX_FAIL " + tag + " " + e.getMessage());
    }
    return Double.NaN;
  }

  private static void createPlot(Model m, String pg, String dset, String label, String colorTable, double maxMises) {
    try { m.result().remove(pg); } catch (Exception ignored) {}
    m.result().create(pg, "PlotGroup3D");
    m.result(pg).label(label);
    m.result(pg).set("data", dset);

    m.result(pg).create("surf1", "Surface");
    m.result(pg).feature("surf1").set("expr", "solid.mises");
    m.result(pg).feature("surf1").set("unit", "Pa");
    m.result(pg).feature("surf1").set("descr", "Von Mises stress");
    try { m.result(pg).feature("surf1").set("colortable", colorTable); } catch (Exception ignored) {}
    try { m.result(pg).feature("surf1").set("rangecoloractive", "on"); } catch (Exception ignored) {}
    try { m.result(pg).feature("surf1").set("rangecolormin", "0[Pa]"); } catch (Exception ignored) {}
    if (!Double.isNaN(maxMises) && maxMises > 0.0) {
      try { m.result(pg).feature("surf1").set("rangecolormax", maxMises + "[Pa]"); } catch (Exception ignored) {}
    }
    try { m.result(pg).feature("surf1").selection().all(); } catch (Exception ignored) {}

    m.result(pg).run();
  }

  private static String exportPng(Model m, String tag, String pg, String filePath) {
    try { m.result().export().remove(tag); } catch (Exception ignored) {}
    try {
      m.result().export().create(tag, "Image3D");
      m.result().export(tag).set("plotgroup", pg);
      try { m.result().export(tag).set("imagetype", "png"); } catch (Exception ignored) {}
      try { m.result().export(tag).set("qualitylevel", "95"); } catch (Exception ignored) {}
      try { m.result().export(tag).set("unit", "px"); } catch (Exception ignored) {}
      try { m.result().export(tag).set("width", 1400); } catch (Exception ignored) {}
      try { m.result().export(tag).set("height", 980); } catch (Exception ignored) {}
      m.result().export(tag).set("pngfilename", filePath);
      m.result().export(tag).run();
      return filePath;
    } catch (Exception e) {
      System.out.println("EXPORT_FAIL " + tag + " " + e.getMessage());
      return "";
    }
  }

  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", MPH); }
    catch (IOException e) { throw new RuntimeException("Failed to load model", e); }

    LinkedHashMap<String, String> dsetsVm = new LinkedHashMap<String, String>();
    dsetsVm.put("std1", "dset6");
    dsetsVm.put("std_nh", "dset1");
    dsetsVm.put("std_og", "dset2");
    dsetsVm.put("std_pr", "dset5");

    LinkedHashMap<String, String> dsetsMr = new LinkedHashMap<String, String>();
    dsetsMr.put("std_mr2", "dset3");
    dsetsMr.put("std_mr5", "dset4");

    for (Map.Entry<String, String> e : dsetsVm.entrySet()) {
      String study = e.getKey();
      String dset = e.getValue();
      double mx = evalMax(m, "mx_contrast_" + study, dset);
      String pg = "pg_contrast_vm_" + study;
      createPlot(m, pg, dset, "Von Mises " + study + " (High-Contrast " + COLOR_VM + ")", COLOR_VM, mx);
      String out = OUT_DIR + "/" + study + "_von_mises_contrast.png";
      exportPng(m, "img_contrast_" + study, pg, out);
      System.out.println("EXPORT_ROW,VM," + study + "," + COLOR_VM + "," + mx + "," + out);
    }

    for (Map.Entry<String, String> e : dsetsMr.entrySet()) {
      String study = e.getKey();
      String dset = e.getValue();
      double mx = evalMax(m, "mx_contrast_" + study, dset);
      String pg = "pg_contrast_mr_" + study;
      createPlot(m, pg, dset, "Mooney-Rivlin " + study + " (High-Contrast " + COLOR_MR + ")", COLOR_MR, mx);
      String out = OUT_DIR + "/" + study + "_mooney_rivlin_contrast.png";
      exportPng(m, "img_contrast_" + study, pg, out);
      System.out.println("EXPORT_ROW,MR," + study + "," + COLOR_MR + "," + mx + "," + out);
    }

    try { m.save(MPH); } catch (IOException ignored) {}
    System.out.println("ExportContrastSeparatedViews done");
  }
}

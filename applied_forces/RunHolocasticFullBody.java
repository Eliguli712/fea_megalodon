import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

public class RunHolocasticFullBody {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
  private static final String MSH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol.msh";
  // Local hotspot crop to mitigate pressure-step inverted element warning near (4.05427, 3.20627, 19.92).
  private static final String BDF_CONFORMING = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol_noperson2_fixinv1.bdf";
  private static final String EXPORT_DIR = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/exports";
  private static final String IMG_SUBDIR = "holocastic_full_body_images";
  private static final String HTML_NAME = "holocastic_full_body_results.html";
  private static final String COLOR_TABLE_VON_MISES = "Turbo";
  private static final String COLOR_TABLE_MOONEY = "Prism";

  private static void p(String s) { System.out.println(s); }

  private static boolean hasStudy(Model m, String tag) {
    try { m.study(tag); return true; } catch (Exception e) { return false; }
  }

  private static boolean hasSolidFeature(Model m, String tag) {
    try { m.component("comp1").physics("solid").feature(tag); return true; } catch (Exception e) { return false; }
  }

  private static void safeSetSolid(Model m, String feat, String key, String val) {
    try { m.component("comp1").physics("solid").feature(feat).set(key, val); } catch (Exception ignored) {}
  }

  private static void safeSetSolidVec(Model m, String feat, String key, String[] val) {
    try { m.component("comp1").physics("solid").feature(feat).set(key, val); } catch (Exception ignored) {}
  }

  private static void safeActivateSolid(Model m, String feat, boolean on) {
    try { m.component("comp1").physics("solid").feature(feat).active(on); } catch (Exception ignored) {}
  }

  private static void configureMeshFromMsh(Model m) {
    // Ensure malformed torus stays removed and stale part input tokens are numeric.
    try { m.geom("part1").feature().remove("tor1"); } catch (Exception ignored) {}
    try { m.geom("part1").inputParam().set("solid", "0"); } catch (Exception ignored) {}
    try { m.geom("part1").inputParam().set("endsolid", "0"); } catch (Exception ignored) {}

    try { m.component("comp1").mesh("mesh1").feature().remove("impmsh"); } catch (Exception ignored) {}
    try { m.component("comp1").mesh("mesh1").feature().remove("fin"); } catch (Exception ignored) {}
    m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");

    // Use whole-body volumetric tetra mesh derived from __tracked_surface_tet_vol.msh.
    m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "nastran");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("filename", BDF_CONFORMING);

    // Build import feature only (full mesh-sequence finalize can fail with stale feature chains).
    m.component("comp1").mesh("mesh1").run("impmsh");

    // Explicitly build volumetric mesh part 2 from the imported body for study solves.
    try {
      m.component("comp1").mesh("mesh2").feature("size1").selection().geom("geom1", 3);
      m.component("comp1").mesh("mesh2").feature("size1").selection().all();
    } catch (Exception ignored) {}
    try {
      m.component("comp1").mesh("mesh2").feature("ftet1").selection().geom("geom1", 3);
      m.component("comp1").mesh("mesh2").feature("ftet1").selection().all();
    } catch (Exception ignored) {}
    m.component("comp1").mesh("mesh2").run();

    for (String st : new String[]{"std1","std_nh","std_og","std_mr2","std_mr5","std_pr"}) {
      if (hasStudy(m, st)) {
        try { m.study(st).feature("stat").set("mesh", new String[][]{{"geom1","mesh2"}}); } catch (Exception ignored) {}
        try { m.study(st).feature("stat").set("plot", "off"); } catch (Exception ignored) {}
        try { m.study(st).feature("stat").set("geometricNonlinearity", "off"); } catch (Exception ignored) {}
      }
    }
  }

  private static void configureWholeBodyPhysics(Model m) {
    m.param().set("thrust_body", "2e4[N/m^3]");
    m.param().descr("thrust_body", "Whole-body volumetric thrust load.");
    m.param().set("pressure_global", "2e3[Pa]");
    m.param().descr("pressure_global", "Global pressure load for pressure study.");

    m.param().set("kappa_bulk", "2.5e8[Pa]");
    m.param().set("mu_ref", "2.5e7[Pa]");
    m.param().set("lambda_ref", "kappa_bulk-2*mu_ref/3");
    m.param().set("ogden_mu1", "2.2e7[Pa]");
    m.param().set("ogden_alpha1", "1.3");
    m.param().set("mr2_c10", "1.6e7[Pa]");
    m.param().set("mr2_c01", "4.0e6[Pa]");
    m.param().set("mr5_c10", "1.2e7[Pa]");
    m.param().set("mr5_c01", "3.0e6[Pa]");
    m.param().set("mr5_c20", "2.0e6[Pa]");
    m.param().set("mr5_c11", "1.5e6[Pa]");
    m.param().set("mr5_c02", "8.0e5[Pa]");

    // Treat full tracked body (surface + interior tets) as one holocastic solid entity.
    m.component("comp1").physics("solid").selection().all();

    // Constrain all boundaries.
    if (!hasSolidFeature(m, "fix1")) {
      m.component("comp1").physics("solid").create("fix1", "Fixed", 2);
    }
    m.component("comp1").physics("solid").feature("fix1").selection().geom("geom1", 2);
    m.component("comp1").physics("solid").feature("fix1").selection().all();
    safeActivateSolid(m, "fix1", true);

    // Constrain all edges (external + internal edges represented in geometry topology).
    if (!hasSolidFeature(m, "fixe_all")) {
      m.component("comp1").physics("solid").create("fixe_all", "Fixed", 1);
    }
    m.component("comp1").physics("solid").feature("fixe_all").selection().geom("geom1", 1);
    m.component("comp1").physics("solid").feature("fixe_all").selection().all();
    safeActivateSolid(m, "fixe_all", true);

    // Whole-body volumetric loading.
    if (!hasSolidFeature(m, "bodyall")) {
      m.component("comp1").physics("solid").create("bodyall", "BodyLoad", 3);
    }
    m.component("comp1").physics("solid").feature("bodyall").selection().geom("geom1", 3);
    m.component("comp1").physics("solid").feature("bodyall").selection().all();
    safeSetSolidVec(m, "bodyall", "FperVol", new String[]{"0", "0", "thrust_body"});
    safeActivateSolid(m, "bodyall", true);

    // Keep pressure load available for dedicated pressure study, but not snout-restricted.
    if (hasSolidFeature(m, "bndl_pr")) {
      m.component("comp1").physics("solid").feature("bndl_pr").selection().geom("geom1", 2);
      m.component("comp1").physics("solid").feature("bndl_pr").selection().all();
      safeSetSolid(m, "bndl_pr", "forceType", "FollowerPressure");
      safeSetSolid(m, "bndl_pr", "pressure", "pressure_global");
    }

    // Disable snout-only force feature to loosen prior boundary restrictions.
    safeActivateSolid(m, "bndl1", false);
    safeActivateSolid(m, "rms1", false);

    // Material model scopes and parameters on full body.
    for (String feat : new String[]{"lemm1","hmm_nh","hmm_og","hmm_mr2","hmm_mr5"}) {
      if (hasSolidFeature(m, feat)) {
        try { m.component("comp1").physics("solid").feature(feat).selection().geom("geom1", 3); } catch (Exception ignored) {}
        try { m.component("comp1").physics("solid").feature(feat).selection().all(); } catch (Exception ignored) {}
      }
    }

    safeSetSolid(m, "lemm1", "E_mat", "userdef");
    safeSetSolid(m, "lemm1", "E", "1.5e8[Pa]");
    safeSetSolid(m, "lemm1", "nu_mat", "userdef");
    safeSetSolid(m, "lemm1", "nu", "0.3");
    safeSetSolid(m, "lemm1", "rho_mat", "userdef");
    safeSetSolid(m, "lemm1", "rho", "1100[kg/m^3]");

    safeSetSolid(m, "hmm_nh", "MaterialModel", "NeoHookean");
    safeSetSolid(m, "hmm_nh", "muLame_mat", "userdef");
    safeSetSolid(m, "hmm_nh", "muLame", "mu_ref");
    safeSetSolid(m, "hmm_nh", "lambLame_mat", "userdef");
    safeSetSolid(m, "hmm_nh", "lambLame", "lambda_ref");
    safeSetSolid(m, "hmm_nh", "K2_mat", "userdef");
    safeSetSolid(m, "hmm_nh", "K2", "kappa_bulk");
    safeSetSolid(m, "hmm_nh", "K3_mat", "userdef");
    safeSetSolid(m, "hmm_nh", "K3", "0[Pa]");
    safeSetSolid(m, "hmm_nh", "G_mat", "userdef");
    safeSetSolid(m, "hmm_nh", "G", "mu_ref");
    safeSetSolid(m, "hmm_nh", "K_mat", "userdef");
    safeSetSolid(m, "hmm_nh", "K", "kappa_bulk");

    safeSetSolid(m, "hmm_og", "MaterialModel", "Ogden");
    safeSetSolid(m, "hmm_og", "muLame_mat", "userdef");
    safeSetSolid(m, "hmm_og", "muLame", "mu_ref");
    safeSetSolid(m, "hmm_og", "lambLame_mat", "userdef");
    safeSetSolid(m, "hmm_og", "lambLame", "lambda_ref");
    safeSetSolid(m, "hmm_og", "K2_mat", "userdef");
    safeSetSolid(m, "hmm_og", "K2", "kappa_bulk");
    safeSetSolid(m, "hmm_og", "K3_mat", "userdef");
    safeSetSolid(m, "hmm_og", "K3", "0[Pa]");
    safeSetSolid(m, "hmm_og", "mup", "ogden_mu1");
    safeSetSolid(m, "hmm_og", "alphap", "ogden_alpha1");
    safeSetSolid(m, "hmm_og", "kappa", "kappa_bulk");

    safeSetSolid(m, "hmm_mr2", "MaterialModel", "MooneyRivlin");
    safeSetSolid(m, "hmm_mr2", "muLame_mat", "userdef");
    safeSetSolid(m, "hmm_mr2", "muLame", "mu_ref");
    safeSetSolid(m, "hmm_mr2", "lambLame_mat", "userdef");
    safeSetSolid(m, "hmm_mr2", "lambLame", "lambda_ref");
    safeSetSolid(m, "hmm_mr2", "K2_mat", "userdef");
    safeSetSolid(m, "hmm_mr2", "K2", "kappa_bulk");
    safeSetSolid(m, "hmm_mr2", "K3_mat", "userdef");
    safeSetSolid(m, "hmm_mr2", "K3", "0[Pa]");
    safeSetSolid(m, "hmm_mr2", "C10_mat", "userdef");
    safeSetSolid(m, "hmm_mr2", "C10", "mr2_c10");
    safeSetSolid(m, "hmm_mr2", "C01_mat", "userdef");
    safeSetSolid(m, "hmm_mr2", "C01", "mr2_c01");
    safeSetSolid(m, "hmm_mr2", "kappa", "kappa_bulk");

    safeSetSolid(m, "hmm_mr5", "MaterialModel", "MooneyRivlin5parameters");
    safeSetSolid(m, "hmm_mr5", "muLame_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "muLame", "mu_ref");
    safeSetSolid(m, "hmm_mr5", "lambLame_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "lambLame", "lambda_ref");
    safeSetSolid(m, "hmm_mr5", "K2_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "K2", "kappa_bulk");
    safeSetSolid(m, "hmm_mr5", "K3_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "K3", "0[Pa]");
    safeSetSolid(m, "hmm_mr5", "C10_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "C10", "mr5_c10");
    safeSetSolid(m, "hmm_mr5", "C01_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "C01", "mr5_c01");
    safeSetSolid(m, "hmm_mr5", "C20_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "C20", "mr5_c20");
    safeSetSolid(m, "hmm_mr5", "C11_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "C11", "mr5_c11");
    safeSetSolid(m, "hmm_mr5", "C02_mat", "userdef");
    safeSetSolid(m, "hmm_mr5", "C02", "mr5_c02");
    safeSetSolid(m, "hmm_mr5", "kappa", "kappa_bulk");
  }

  private static void activateCase(Model m, String material, boolean pressureOn) {
    safeActivateSolid(m, "lemm1", "linear".equals(material));
    safeActivateSolid(m, "hmm_nh", "nh".equals(material));
    safeActivateSolid(m, "hmm_og", "og".equals(material));
    safeActivateSolid(m, "hmm_mr2", "mr2".equals(material));
    safeActivateSolid(m, "hmm_mr5", "mr5".equals(material));

    safeActivateSolid(m, "bodyall", true);
    safeActivateSolid(m, "bndl_pr", pressureOn);
    safeActivateSolid(m, "bndl1", false);
  }

  private static double evalMaxMises(Model m, String tag, String dataset) {
    try { m.result().numerical().remove(tag); } catch (Exception ignored) {}
    try {
      m.result().numerical().create(tag, "MaxVolume");
      m.result().numerical(tag).set("expr", new String[]{"solid.mises"});
      m.result().numerical(tag).set("unit", new String[]{"Pa"});
      m.result().numerical(tag).set("data", dataset);
      m.result().numerical(tag).selection().all();
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) return r[0][0];
    } catch (Exception e) {
      p("max eval failed for " + dataset + ": " + e.getMessage());
    }
    return Double.NaN;
  }

  private static void ensurePlot(Model m, String pg, String dataset, String label, double maxMises, String colorTable) {
    try { m.result().remove(pg); } catch (Exception ignored) {}
    m.result().create(pg, "PlotGroup3D");
    m.result(pg).label(label);
    m.result(pg).set("data", dataset);
    m.result(pg).create("surf1", "Surface");
    m.result(pg).feature("surf1").set("expr", "solid.mises");
    m.result(pg).feature("surf1").set("unit", "Pa");
    m.result(pg).feature("surf1").set("descr", "Von Mises stress");
    // Color table explicitly requested.
    try { m.result(pg).feature("surf1").set("colortable", colorTable); } catch (Exception ignored) {}
    // Force a vivid full-span gradient instead of muted auto-range.
    try { m.result(pg).feature("surf1").set("rangecoloractive", "on"); } catch (Exception ignored) {}
    try { m.result(pg).feature("surf1").set("rangecolormin", "0[Pa]"); } catch (Exception ignored) {}
    if (!Double.isNaN(maxMises) && maxMises > 0.0) {
      try { m.result(pg).feature("surf1").set("rangecolormax", maxMises + "[Pa]"); } catch (Exception ignored) {}
    }
    try { m.result(pg).feature("surf1").selection().all(); } catch (Exception ignored) {}
    m.result(pg).run();
  }

  private static String exportImage(Model m, String exportTag, String plotGroup, String filePath) {
    try { m.result().export().remove(exportTag); } catch (Exception ignored) {}
    try {
      m.result().export().create(exportTag, "Image3D");
      m.result().export(exportTag).set("plotgroup", plotGroup);
      try { m.result().export(exportTag).set("imagetype", "png"); } catch (Exception ignored) {}
      try { m.result().export(exportTag).set("qualitylevel", "92"); } catch (Exception ignored) {}
      try { m.result().export(exportTag).set("unit", "px"); } catch (Exception ignored) {}
      try { m.result().export(exportTag).set("width", 1280); } catch (Exception ignored) {}
      try { m.result().export(exportTag).set("height", 960); } catch (Exception ignored) {}
      m.result().export(exportTag).set("pngfilename", filePath);
      m.result().export(exportTag).run();
      return filePath;
    } catch (Exception e) {
      p("Image export failed for " + plotGroup + ": " + e.getMessage());
      return "";
    }
  }

  private static String datasetForStudy(String studyTag) {
    if ("std1".equals(studyTag)) return "dset6";
    if ("std_nh".equals(studyTag)) return "dset1";
    if ("std_og".equals(studyTag)) return "dset2";
    if ("std_mr2".equals(studyTag)) return "dset3";
    if ("std_mr5".equals(studyTag)) return "dset4";
    if ("std_pr".equals(studyTag)) return "dset5";
    return "dset6";
  }

  private static String materialForStudy(String studyTag) {
    if ("std1".equals(studyTag)) return "linear";
    if ("std_nh".equals(studyTag)) return "nh";
    if ("std_og".equals(studyTag)) return "og";
    if ("std_mr2".equals(studyTag)) return "mr2";
    if ("std_mr5".equals(studyTag)) return "mr5";
    if ("std_pr".equals(studyTag)) return "mr5";
    return "linear";
  }

  public static void main(String[] args) {
    int ctCount = 6123;
    int ctMaxId = 20282;
    p("CTETRA count=" + ctCount + " maxID=" + ctMaxId);

    Model m;
    try { m = ModelUtil.load("Model", MPH); }
    catch (IOException e) { throw new RuntimeException("Failed to load model", e); }

    try {
      configureMeshFromMsh(m);
      configureWholeBodyPhysics(m);
    } catch (Exception e) {
      throw new RuntimeException("Model configuration failed: " + e.getMessage(), e);
    }

    int domCount = -1, bndCount = -1, edgCount = -1;
    try {
      m.component("comp1").physics("solid").selection().all();
      domCount = m.component("comp1").physics("solid").selection().entities(3).length;
    } catch (Exception ignored) {}
    try {
      m.component("comp1").physics("solid").feature("fix1").selection().all();
      bndCount = m.component("comp1").physics("solid").feature("fix1").selection().entities(2).length;
    } catch (Exception ignored) {}
    try {
      m.component("comp1").physics("solid").feature("fixe_all").selection().all();
      edgCount = m.component("comp1").physics("solid").feature("fixe_all").selection().entities(1).length;
    } catch (Exception ignored) {}
    p("counts: dom=" + domCount + " bnd=" + bndCount + " edge=" + edgCount);

    String imgDir = EXPORT_DIR + "/" + IMG_SUBDIR;

    String[] studies = new String[]{"std1","std_nh","std_og","std_mr2","std_mr5","std_pr"};
    Map<String, String> status = new LinkedHashMap<String, String>();
    Map<String, Double> maxByStudy = new LinkedHashMap<String, Double>();
    Map<String, String> relImageByStudy = new LinkedHashMap<String, String>();

    for (String st : studies) {
      if (!hasStudy(m, st)) {
        status.put(st, "missing");
        maxByStudy.put(st, Double.NaN);
        relImageByStudy.put(st, "");
        continue;
      }

      boolean pressureOn = "std_pr".equals(st);
      activateCase(m, materialForStudy(st), pressureOn);

      try {
        m.study(st).run();
        status.put(st, "ok");
      } catch (Exception e) {
        status.put(st, "failed: " + e.getMessage());
      }

      String dset = datasetForStudy(st);
      double mx = evalMaxMises(m, "mx_" + st, dset);
      maxByStudy.put(st, mx);
      p(st + " max mises=" + mx);

      boolean isMooney = "std_mr2".equals(st) || "std_mr5".equals(st);
      String pg = "pg_holo_" + st;
      String colorTable = isMooney ? COLOR_TABLE_MOONEY : COLOR_TABLE_VON_MISES;
      try {
        ensurePlot(m, pg, dset, "Von Mises Cloud " + st + " (Holocastic Full Body)", mx, colorTable);
        String imgName = isMooney ? (st + "_mooney_rivlin.png") : (st + "_von_mises.png");
        String imgPath = imgDir + "/" + imgName;
        String exported = exportImage(m, "img_" + st, pg, imgPath);
        if (exported != null && !exported.isEmpty()) {
          relImageByStudy.put(st, IMG_SUBDIR + "/" + imgName);
          p("IMAGE " + st + " " + imgPath);
        } else {
          relImageByStudy.put(st, "");
        }
      } catch (Exception e) {
        relImageByStudy.put(st, "");
        p("plot/export failed for " + st + ": " + e.getMessage());
      }
    }

    // Keep an always-on summary cloud as a default result view.
    try {
      ensurePlot(
        m,
        "pg_vms_holocastic",
        "dset6",
        "Von Mises Stress Cloud (Holocastic Full Body)",
        maxByStudy.get("std1"),
        COLOR_TABLE_VON_MISES
      );
    } catch (Exception ignored) {}

    // Save model and print machine-readable summary for external HTML assembly.
    try { m.save(MPH); }
    catch (IOException e) { throw new RuntimeException("Failed to save model", e); }

    p("SUMMARY dom=" + domCount + " bnd=" + bndCount + " edge=" + edgCount + " ctetra_count=" + ctCount + " ctetra_maxid=" + ctMaxId);
    for (String st : studies) {
      p("STUDY " + st + " status=" + status.get(st) + " max_mises=" + maxByStudy.get(st));
    }

    p("Done.");
  }
}

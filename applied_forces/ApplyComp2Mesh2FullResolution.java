import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ApplyComp2Mesh2FullResolution {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
  // Prefer the highest-resolution raw full-body volumetric mesh first.
  private static final String BDF_HIRES_RAW_PRIMARY =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/tetwild_input_smoothed_tet_vol.bdf";
  private static final String BDF_HIRES_RAW_SECONDARY =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/tetwild_input_smoothed_comsol_tet_vol.bdf";
  private static final String BDF_HIRES_RAW_FALLBACK =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol.bdf";

  private static String COMP2_GEOM = "geom2";
  private static String COMP2_MESH_IMPORT = "mesh3";
  private static String COMP2_MESH_SOLVE = "mesh4";
  private static String COMP2_SOLID = "solid2";
  private static String RAW_BDF_IN_USE = BDF_HIRES_RAW_PRIMARY;
  private static final String[] RAW_BDF_CANDIDATES =
      new String[] {BDF_HIRES_RAW_PRIMARY, BDF_HIRES_RAW_SECONDARY, BDF_HIRES_RAW_FALLBACK};

  private static boolean hasStudy(Model m, String tag) {
    try {
      m.study(tag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean hasSolidFeature(Model m, String tag) {
    try {
      m.component("comp2").physics(COMP2_SOLID).feature(tag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static void safeSetSolid(Model m, String feat, String key, String val) {
    try {
      m.component("comp2").physics(COMP2_SOLID).feature(feat).set(key, val);
    } catch (Exception ignored) {
    }
  }

  private static void safeSetSolidVec(Model m, String feat, String key, String[] val) {
    try {
      m.component("comp2").physics(COMP2_SOLID).feature(feat).set(key, val);
    } catch (Exception ignored) {
    }
  }

  private static void safeActivateSolid(Model m, String feat, boolean on) {
    try {
      m.component("comp2").physics(COMP2_SOLID).feature(feat).active(on);
    } catch (Exception ignored) {
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

  private static void refreshComp2Tags(Model m) {
    try {
      String[] geoms = m.component("comp2").geom().tags();
      if (geoms != null && geoms.length > 0) COMP2_GEOM = geoms[0];
    } catch (Exception ignored) {
    }

    try {
      String[] meshes = m.component("comp2").mesh().tags();
      if (meshes != null && meshes.length > 0) {
        for (String mt : meshes) {
          try {
            m.component("comp2").mesh(mt).feature("impmsh");
            COMP2_MESH_IMPORT = mt;
          } catch (Exception ignored) {
          }
          try {
            m.component("comp2").mesh(mt).feature("ftet1");
            COMP2_MESH_SOLVE = mt;
          } catch (Exception ignored) {
          }
        }
        if (COMP2_MESH_IMPORT == null || COMP2_MESH_IMPORT.isEmpty()) {
          COMP2_MESH_IMPORT = meshes[0];
        }
        if (COMP2_MESH_SOLVE == null || COMP2_MESH_SOLVE.isEmpty()) {
          COMP2_MESH_SOLVE = meshes[Math.min(1, meshes.length - 1)];
        }
      }
    } catch (Exception ignored) {
    }

    try {
      String[] phys = m.component("comp2").physics().tags();
      if (phys != null && phys.length > 0) {
        COMP2_SOLID = phys[0];
      }
    } catch (Exception ignored) {
    }
  }

  private static void rebuildComp2FromComp1(Model m, List<String> logs) {
    try {
      m.component().remove("comp2");
      logs.add("COMP2_REMOVE|ok=true");
    } catch (Exception e) {
      logs.add("COMP2_REMOVE|ok=false|err=" + e.getMessage());
    }

    try {
      m.component().copy("comp2", "comp1");
      logs.add("COMP2_COPY|ok=true");
    } catch (Exception e) {
      throw new RuntimeException("Failed to copy comp1 to comp2: " + e.getMessage(), e);
    }

    refreshComp2Tags(m);
    logs.add(
        "COMP2_TAGS|geom="
            + COMP2_GEOM
            + "|mesh_import="
            + COMP2_MESH_IMPORT
            + "|mesh_solve="
            + COMP2_MESH_SOLVE
            + "|solid="
            + COMP2_SOLID);
  }

  private static void configureComp2Mesh(Model m, List<String> logs) {
    boolean meshReady = false;
    try {
      m.component("comp2").mesh(COMP2_MESH_IMPORT).feature("impmsh").set("source", "nastran");
    } catch (Exception ignored) {
    }

    for (String candidate : RAW_BDF_CANDIDATES) {
      try {
        m.component("comp2").mesh(COMP2_MESH_IMPORT).feature("impmsh").set("filename", candidate);
        m.component("comp2").mesh(COMP2_MESH_IMPORT).run("impmsh");
        logs.add("COMP2_MESH_IMPORT|ok=true|path=" + candidate);
      } catch (Exception e) {
        logs.add("COMP2_MESH_IMPORT_ATTEMPT|path=" + candidate + "|err=" + e.getMessage());
        continue;
      }

      try {
        m.component("comp2")
            .mesh(COMP2_MESH_SOLVE)
            .feature("size1")
            .selection()
            .geom(COMP2_GEOM, 3);
        m.component("comp2").mesh(COMP2_MESH_SOLVE).feature("size1").selection().all();
      } catch (Exception ignored) {
      }
      try {
        m.component("comp2")
            .mesh(COMP2_MESH_SOLVE)
            .feature("ftet1")
            .selection()
            .geom(COMP2_GEOM, 3);
        m.component("comp2").mesh(COMP2_MESH_SOLVE).feature("ftet1").selection().all();
      } catch (Exception ignored) {
      }

      try {
        m.component("comp2").mesh(COMP2_MESH_SOLVE).run();
        int dom = m.component("comp2").mesh(COMP2_MESH_SOLVE).getNumElem("tet");
        logs.add("COMP2_MESH2_RUN|ok=true|path=" + candidate);
        logs.add("COMP2_MESH2_TET_COUNT|" + dom);
        if (dom > 0) {
          RAW_BDF_IN_USE = candidate;
          logs.add("RAW_BDF|path=" + RAW_BDF_IN_USE);
          meshReady = true;
          break;
        }
      } catch (Exception e) {
        logs.add("COMP2_MESH2_RUN|ok=false|path=" + candidate + "|err=" + e.getMessage());
      }
    }

    if (!meshReady) {
      logs.add("COMP2_MESH2_TET_COUNT|0");
      throw new RuntimeException("All high-resolution raw BDF candidates failed mesh2 generation.");
    }
  }

  private static void configureComp2Physics(Model m) {
    m.param().set("thrust_body", "2e4[N/m^3]");
    m.param().set("pressure_global", "2e3[Pa]");
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

    m.component("comp2").physics(COMP2_SOLID).selection().all();

    if (!hasSolidFeature(m, "fix1")) {
      m.component("comp2").physics(COMP2_SOLID).create("fix1", "Fixed", 2);
    }
    m.component("comp2").physics(COMP2_SOLID).feature("fix1").selection().geom(COMP2_GEOM, 2);
    m.component("comp2").physics(COMP2_SOLID).feature("fix1").selection().all();
    safeActivateSolid(m, "fix1", true);

    if (!hasSolidFeature(m, "fixe_all")) {
      m.component("comp2").physics(COMP2_SOLID).create("fixe_all", "Fixed", 1);
    }
    m.component("comp2")
        .physics(COMP2_SOLID)
        .feature("fixe_all")
        .selection()
        .geom(COMP2_GEOM, 1);
    m.component("comp2").physics(COMP2_SOLID).feature("fixe_all").selection().all();
    safeActivateSolid(m, "fixe_all", true);

    if (!hasSolidFeature(m, "bodyall")) {
      m.component("comp2").physics(COMP2_SOLID).create("bodyall", "BodyLoad", 3);
    }
    m.component("comp2").physics(COMP2_SOLID).feature("bodyall").selection().geom(COMP2_GEOM, 3);
    m.component("comp2").physics(COMP2_SOLID).feature("bodyall").selection().all();
    safeSetSolidVec(m, "bodyall", "FperVol", new String[] {"0", "0", "thrust_body"});
    safeActivateSolid(m, "bodyall", true);

    if (hasSolidFeature(m, "bndl_pr")) {
      m.component("comp2").physics(COMP2_SOLID).feature("bndl_pr").selection().geom(COMP2_GEOM, 2);
      m.component("comp2").physics(COMP2_SOLID).feature("bndl_pr").selection().all();
      safeSetSolid(m, "bndl_pr", "forceType", "FollowerPressure");
      safeSetSolid(m, "bndl_pr", "pressure", "pressure_global");
    }

    for (String feat : new String[] {"lemm1", "hmm_nh", "hmm_og", "hmm_mr2", "hmm_mr5"}) {
      if (hasSolidFeature(m, feat)) {
        try {
          m.component("comp2")
              .physics(COMP2_SOLID)
              .feature(feat)
              .selection()
              .geom(COMP2_GEOM, 3);
        } catch (Exception ignored) {
        }
        try {
          m.component("comp2").physics(COMP2_SOLID).feature(feat).selection().all();
        } catch (Exception ignored) {
        }
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

    safeSetSolid(m, "hmm_og", "MaterialModel", "Ogden");
    safeSetSolid(m, "hmm_og", "muLame_mat", "userdef");
    safeSetSolid(m, "hmm_og", "muLame", "mu_ref");
    safeSetSolid(m, "hmm_og", "lambLame_mat", "userdef");
    safeSetSolid(m, "hmm_og", "lambLame", "lambda_ref");
    safeSetSolid(m, "hmm_og", "mup", "ogden_mu1");
    safeSetSolid(m, "hmm_og", "alphap", "ogden_alpha1");
    safeSetSolid(m, "hmm_og", "kappa", "kappa_bulk");

    safeSetSolid(m, "hmm_mr2", "MaterialModel", "MooneyRivlin");
    safeSetSolid(m, "hmm_mr2", "C10_mat", "userdef");
    safeSetSolid(m, "hmm_mr2", "C10", "mr2_c10");
    safeSetSolid(m, "hmm_mr2", "C01_mat", "userdef");
    safeSetSolid(m, "hmm_mr2", "C01", "mr2_c01");
    safeSetSolid(m, "hmm_mr2", "kappa", "kappa_bulk");

    safeSetSolid(m, "hmm_mr5", "MaterialModel", "MooneyRivlin5parameters");
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

    safeActivateSolid(m, "bndl1", false);
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

  private static void configureStudies(Model m, List<String> logs) {
    String[] studies = new String[] {"std1", "std_nh", "std_og", "std_mr2", "std_mr5", "std_pr"};
    for (String st : studies) {
      if (!hasStudy(m, st)) {
        logs.add("STUDY_BIND|" + st + "|status=missing");
        continue;
      }
      try {
        m.study(st)
            .feature("stat")
            .set(
                "mesh",
                new String[][] {
                  {"geom1", "nomesh", COMP2_GEOM, COMP2_MESH_SOLVE}
                });
      } catch (Exception e1) {
        try {
          m.study(st).feature("stat").set("mesh", new String[][] {{COMP2_GEOM, COMP2_MESH_SOLVE}});
        } catch (Exception e2) {
          logs.add("STUDY_BIND|" + st + "|mesh_set_fail=" + e2.getMessage());
        }
      }
      try {
        m.study(st).feature("stat").activate("solid", false);
      } catch (Exception ignored) {
      }
      try {
        m.study(st).feature("stat").activate(COMP2_SOLID, true);
      } catch (Exception ignored) {
      }
      try {
        m.study(st).feature("stat").activate("comp1", false);
      } catch (Exception ignored) {
      }
      try {
        m.study(st).feature("stat").activate("comp2", true);
      } catch (Exception ignored) {
      }
      try {
        m.study(st).feature("stat").set("plot", "off");
      } catch (Exception ignored) {
      }
      logs.add("STUDY_BIND|" + st + "|status=ok");
    }
  }

  private static void configureDatasetsForComp2(Model m, List<String> logs) {
    String[] datasets = new String[] {"dset1", "dset2", "dset3", "dset4", "dset5", "dset6"};
    for (String ds : datasets) {
      try {
        m.result().dataset(ds).set("geom", COMP2_GEOM);
        logs.add("DATASET_GEOM|" + ds + "|geom=" + COMP2_GEOM);
      } catch (Exception e) {
        logs.add("DATASET_GEOM|" + ds + "|set_fail=" + e.getMessage());
      }
    }
  }

  private static double evalMaxMises(Model m, String dataset, String tag) {
    try {
      try {
        m.result().numerical().remove(tag);
      } catch (Exception ignored) {
      }
      m.result().numerical().create(tag, "MaxVolume");
      m.result().numerical(tag).set("expr", new String[] {COMP2_SOLID + ".mises"});
      m.result().numerical(tag).set("unit", new String[] {"Pa"});
      m.result().numerical(tag).set("data", dataset);
      try {
        m.result().numerical(tag).selection().geom(COMP2_GEOM, 3);
      } catch (Exception ignored) {
      }
      m.result().numerical(tag).selection().all();
      m.result().numerical(tag).setResult();
      double[][] r = m.result().numerical(tag).getReal();
      if (r != null && r.length > 0 && r[0].length > 0) {
        double v = r[0][0];
        if (!Double.isInfinite(v) && !Double.isNaN(v)) {
          return v;
        }
      }
    } catch (Exception ignored) {
    }
    // Fallback on boundary maximum if volume selection is unavailable in the result dataset.
    try {
      String btag = tag + "_bnd";
      try {
        m.result().numerical().remove(btag);
      } catch (Exception ignored) {
      }
      m.result().numerical().create(btag, "MaxSurface");
      m.result().numerical(btag).set("expr", new String[] {COMP2_SOLID + ".mises"});
      m.result().numerical(btag).set("unit", new String[] {"Pa"});
      m.result().numerical(btag).set("data", dataset);
      try {
        m.result().numerical(btag).selection().geom(COMP2_GEOM, 2);
      } catch (Exception ignored) {
      }
      m.result().numerical(btag).selection().all();
      m.result().numerical(btag).setResult();
      double[][] rb = m.result().numerical(btag).getReal();
      if (rb != null && rb.length > 0 && rb[0].length > 0) {
        double vb = rb[0][0];
        if (!Double.isInfinite(vb) && !Double.isNaN(vb)) {
          return vb;
        }
      }
    } catch (Exception ignored) {
    }
    return Double.NaN;
  }

  private static void ensureStudyPlot(Model m, String studyTag, String dataset) {
    String pg = "pg_comp2_" + studyTag;
    try {
      m.result().remove(pg);
    } catch (Exception ignored) {
    }
    m.result().create(pg, "PlotGroup3D");
    m.result(pg).label("Von Mises Cloud " + studyTag + " (comp2 mesh2 full-resolution)");
    m.result(pg).set("data", dataset);
    m.result(pg).create("surf1", "Surface");
    m.result(pg).feature("surf1").set("expr", COMP2_SOLID + ".mises");
    m.result(pg).feature("surf1").set("unit", "Pa");
    m.result(pg).feature("surf1").set("descr", "Von Mises stress");
    try {
      m.result(pg).feature("surf1").selection().geom(COMP2_GEOM, 2);
    } catch (Exception ignored) {
    }
    try {
      m.result(pg).feature("surf1").selection().all();
    } catch (Exception ignored) {
    }
    m.result(pg).run();
  }

  public static void main(String[] args) {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed loading model: " + MPH, e);
    }

    List<String> logs = new ArrayList<String>();
    logs.add("MODEL|" + MPH);
    logs.add("RAW_DATA_COMPRESSION|off");
    logs.add("CHUNK_POLICY|study-by-study sequential solve");

    rebuildComp2FromComp1(m, logs);
    configureComp2Mesh(m, logs);
    configureComp2Physics(m);
    configureStudies(m, logs);
    configureDatasetsForComp2(m, logs);

    String[] studies = new String[] {"std1", "std_nh", "std_og", "std_mr2", "std_mr5", "std_pr"};
    for (String st : studies) {
      if (!hasStudy(m, st)) {
        logs.add("STUDY|" + st + "|status=missing");
        continue;
      }
      boolean pressureOn = "std_pr".equals(st);
      activateCase(m, materialForStudy(st), pressureOn);
      try {
        m.study(st).run();
        String ds = datasetForStudy(st);
        double mx = evalMaxMises(m, ds, "mx_comp2_" + st);
        logs.add("STUDY|" + st + "|status=ok|dataset=" + ds + "|max_mises=" + mx);
        try {
          ensureStudyPlot(m, st, ds);
          logs.add("PLOT|" + st + "|expr=" + COMP2_SOLID + ".mises");
        } catch (Exception e) {
          logs.add("PLOT|" + st + "|status=fail|err=" + e.getMessage());
        }
      } catch (Exception e) {
        logs.add("STUDY|" + st + "|status=fail|err=" + e.getMessage());
      }
      // Encourage heap reclamation between long high-resolution study chunks.
      try {
        System.gc();
      } catch (Exception ignored) {
      }
    }

    for (String st : studies) {
      if (!hasStudy(m, st)) continue;
      try {
        String[][] meshMap = m.study(st).feature("stat").getStringMatrix("mesh");
        logs.add("MESHMAP|" + st + "|" + Arrays.deepToString(meshMap));
      } catch (Exception e) {
        logs.add("MESHMAP|" + st + "|read_fail=" + e.getMessage());
      }
      try {
        String[][] act = m.study(st).feature("stat").getStringMatrix("activate");
        logs.add("ACTIVATE|" + st + "|" + Arrays.deepToString(act));
      } catch (Exception e) {
        logs.add("ACTIVATE|" + st + "|read_fail=" + e.getMessage());
      }
    }

    try {
      m.save(MPH);
      logs.add("SAVED|" + MPH);
    } catch (IOException e) {
      logs.add("SAVE_FAIL|" + e.getMessage());
    }

    for (String line : logs) {
      System.out.println(line);
    }
  }
}

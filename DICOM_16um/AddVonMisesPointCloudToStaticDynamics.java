import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.lang.reflect.Method;

public class AddVonMisesPointCloudToStaticDynamics {
  private static final String MODEL_PATH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/static_dynamics.mph";
  private static final String IMAGE_DIR =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports";

  // name, shortTag, preflightBdf, solverBdf
  private static final String[][] ENTITIES = new String[][]{
      {
          "surface_mesh_smoothed",
          "smoothed",
          "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/surface_mesh_smoothed.bdf",
          "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/surface_mesh_smoothed.bdf"
      },
      {
          "tooth_surface_uncompressed",
          "uncompressed",
          "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_uncompressed.bdf",
          "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_uncompressed.bdf"
      },
      {
          "tooth_surface_comsol_tet_vol",
          "rawtet",
          "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_comsol_tet_vol.bdf",
          "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/DICOM_16um/exports/tooth_surface_comsol_tet_vol.bdf"
      }
  };

  private static void safeSet(Object f, String key, String value) {
    try {
      Method m = f.getClass().getMethod("set", String.class, String.class);
      m.invoke(f, key, value);
    } catch (Exception ignored) {
    }
  }

  private static void safeSet(Object f, String key, String[] value) {
    try {
      Method m = f.getClass().getMethod("set", String.class, String[].class);
      m.invoke(f, key, value);
    } catch (Exception ignored) {
    }
  }

  private static void safeSet(Object f, String key, int value) {
    try {
      Method m = f.getClass().getMethod("set", String.class, int.class);
      m.invoke(f, key, value);
    } catch (Exception ignored) {
    }
  }

  private static void safeSet(Object f, String key, double value) {
    try {
      Method m = f.getClass().getMethod("set", String.class, double.class);
      m.invoke(f, key, value);
    } catch (Exception ignored) {
    }
  }

  private static void safeSet(Object f, String key, boolean value) {
    try {
      Method m = f.getClass().getMethod("set", String.class, boolean.class);
      m.invoke(f, key, value);
    } catch (Exception ignored) {
    }
  }

  private static void safeActivate(Model model, String featureTag, boolean active) {
    try {
      model.component("comp1").physics("solid").feature(featureTag).active(active);
    } catch (Exception ignored) {
    }
  }

  private static void clearMesh1Features(Model model) {
    String[] tags = model.component("comp1").mesh("mesh1").feature().tags();
    for (String tag : tags) {
      if ("fin".equals(tag)) {
        continue;
      }
      try {
        model.component("comp1").mesh("mesh1").feature().remove(tag);
      } catch (Exception ignored) {
      }
    }
  }

  private static void clearMpartDeleteFeatures(Model model) {
    String[] tags = model.mesh("mpart1").feature().tags();
    for (String tag : tags) {
      if (tag.startsWith("dele")) {
        try {
          model.mesh("mpart1").feature().remove(tag);
        } catch (Exception ignored) {
        }
      }
    }
  }

  private static boolean preflightOpen(Model model, String entityName, String bdfPath) {
    try {
      model.mesh("mpart1").feature("imp1").set("source", "nastran");
      model.mesh("mpart1").feature("imp1").set("filename", bdfPath);
      model.mesh("mpart1").run("imp1");
      System.out.println("BDF_OPEN|" + entityName + "|" + bdfPath + "|ok=true");
      return true;
    } catch (Exception e) {
      System.out.println("BDF_OPEN|" + entityName + "|" + bdfPath + "|ok=false|" + e.getMessage());
      return false;
    }
  }

  private static void loadBdfForSolve(Model model, String bdfPath) {
    clearMpartDeleteFeatures(model);
    model.mesh("mpart1").feature("imp1").set("source", "nastran");
    model.mesh("mpart1").feature("imp1").set("filename", bdfPath);
    model.mesh("mpart1").feature("imp1").set("createdom", "on");
    model.mesh("mpart1").feature("imp1").set("facepartition", "minimal");
    try {
      model.mesh("mpart1").feature("remf1").selection().all();
    } catch (Exception ignored) {
    }
    model.mesh("mpart1").run();

    clearMesh1Features(model);
    model.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
    model.component("comp1").mesh("mesh1").feature("impmsh").set("source", "sequence");
    model.component("comp1").mesh("mesh1").feature("impmsh").set("sequence", "mpart1");
    model.component("comp1").mesh("mesh1").feature("impmsh").set("buildsource", "on");
    model.component("comp1").mesh("mesh1").feature("impmsh").set("domelemsequence", "on");
    model.component("comp1").mesh("mesh1").feature("impmsh").set("unmesheddom", "on");
    model.component("comp1").mesh("mesh1").run("impmsh");
    model.component("comp1").mesh("mesh1").run("fin");

    try {
      model.study("std_mr5").feature("stat").set("mesh", new String[][]{{"geom1", "mesh1"}});
    } catch (Exception ignored) {
    }
  }

  private static void configureMr5AndLoads(Model model) {
    model.param().set("mr5_c10", "1.2e7[Pa]");
    model.param().set("mr5_c01", "3.0e6[Pa]");
    model.param().set("mr5_c20", "2.0e6[Pa]");
    model.param().set("mr5_c11", "1.5e6[Pa]");
    model.param().set("mr5_c02", "8.0e5[Pa]");
    model.param().set("kappa_bulk", "2.5e8[Pa]");
    model.param().set("force_density_z", "5.0e4[N/m^3]");

    safeActivate(model, "lemm1", false);
    safeActivate(model, "hmm_nh", false);
    safeActivate(model, "hmm_og", false);
    safeActivate(model, "hmm_mr2", false);
    safeActivate(model, "hmm_mr5", true);

    safeActivate(model, "fix1", false);
    safeActivate(model, "fixe_all", false);
    safeActivate(model, "bndl1", false);
    safeActivate(model, "bndl_pr", false);
    safeActivate(model, "bodyall", false);

    Object mr5 = model.component("comp1").physics("solid").feature("hmm_mr5");
    safeSet(mr5, "MaterialModel", "MooneyRivlin5parameters");
    safeSet(mr5, "Compressibility_MooneyRivlin", "NearlyIncompressible");
    safeSet(mr5, "C10_mat", "userdef");
    safeSet(mr5, "C10", "mr5_c10");
    safeSet(mr5, "C01_mat", "userdef");
    safeSet(mr5, "C01", "mr5_c01");
    safeSet(mr5, "C20_mat", "userdef");
    safeSet(mr5, "C20", "mr5_c20");
    safeSet(mr5, "C11_mat", "userdef");
    safeSet(mr5, "C11", "mr5_c11");
    safeSet(mr5, "C02_mat", "userdef");
    safeSet(mr5, "C02", "mr5_c02");
    safeSet(mr5, "kappa", "kappa_bulk");

    try {
      model.component("comp1").physics("solid").prop("ShapeProperty").set("order_displacement", "1");
      model.component("comp1").physics("solid").prop("ShapeProperty").set("order_pressure", "1");
      model.component("comp1").physics("solid").prop("ShapeProperty").set("displacementOrder", "linear");
      model.study("std_mr5").feature("stat").set("shapeorder", "linear");
    } catch (Exception ignored) {
    }

    try {
      model.component("comp1").physics("solid").feature().remove("rmsd1");
    } catch (Exception ignored) {
    }
    try {
      model.component("comp1").physics("solid").feature().remove("bodyd1");
    } catch (Exception ignored) {
    }
    model.component("comp1").physics("solid").create("rmsd1", "RigidMotionSuppression", 3);
    model.component("comp1").physics("solid").feature("rmsd1").selection().all();
    model.component("comp1").physics("solid").create("bodyd1", "BodyLoad", 3);
    model.component("comp1").physics("solid").feature("bodyd1").selection().all();
    safeSet(model.component("comp1").physics("solid").feature("bodyd1"), "F", new String[]{"0", "0", "force_density_z"});
    safeSet(model.component("comp1").physics("solid").feature("bodyd1"), "FperVol", new String[]{"0", "0", "force_density_z"});
  }

  private static void exportVonMisesData(Model model, String shortTag) {
    String exportTag = "dataexp_vm_" + shortTag;
    String outTxt = IMAGE_DIR + "/von_mises_data_" + shortTag + ".txt";
    try {
      model.result().export().remove(exportTag);
    } catch (Exception ignored) {
    }
    model.result().export().create(exportTag, "Data");
    Object ex = model.result().export(exportTag);
    safeSet(ex, "data", "dset4");
    safeSet(ex, "expr", new String[]{"solid.mises"});
    safeSet(ex, "descr", new String[]{"Von Mises stress"});
    safeSet(ex, "location", "fromdataset");
    safeSet(ex, "filename", outTxt);
    safeSet(ex, "header", true);
    safeSet(ex, "fullprec", true);
    model.result().export(exportTag).run();
    System.out.println("VON_MISES_DATA_READY|" + shortTag + "|" + outTxt);
  }

  private static void buildVonMisesPlots(Model model, String shortTag, String entityName) {
    String pgPoint = "pg_vm_point_" + shortTag;
    String pgSurface = "pg_vm_surface_" + shortTag;

    try {
      model.result().remove(pgPoint);
    } catch (Exception ignored) {
    }
    model.result().create(pgPoint, "PlotGroup3D");
    model.result(pgPoint).label("Von Mises Point Cloud - " + entityName);
    safeSet(model.result(pgPoint), "data", "dset4");
    model.result(pgPoint).create("pt1", "Point");
    ResultFeature pt = model.result(pgPoint).feature("pt1");
    try {
      pt.selection().geom("geom1", 2);
      pt.selection().all();
    } catch (Exception ignored) {
    }
    safeSet(pt, "expr", new String[]{"solid.mises"});
    safeSet(pt, "descr", new String[]{"Von Mises stress"});
    safeSet(pt, "coloring", "colortable");
    safeSet(pt, "colortable", "ThermalLight");
    safeSet(pt, "smooth", "internal");
    safeSet(pt, "placement", "meshnodes");
    safeSet(pt, "maxpointcount", 12000);
    safeSet(pt, "pointtype", "point");
    safeSet(pt, "fixedpointsize", true);
    safeSet(pt, "pointradius", 1.2);
    safeSet(pt, "colorlegend", true);
    model.result(pgPoint).run();

    try {
      model.result().remove(pgSurface);
    } catch (Exception ignored) {
    }
    model.result().create(pgSurface, "PlotGroup3D");
    model.result(pgSurface).label("Von Mises Surface - " + entityName);
    safeSet(model.result(pgSurface), "data", "dset4");
    model.result(pgSurface).create("surf1", "Surface");
    ResultFeature sf = model.result(pgSurface).feature("surf1");
    safeSet(sf, "expr", "solid.mises");
    safeSet(sf, "descr", "Von Mises stress");
    safeSet(sf, "colortable", "ThermalLight");
    safeSet(sf, "smooth", "internal");
    safeSet(sf, "colorlegend", true);
    model.result(pgSurface).run();

    System.out.println("VON_MISES_POINT_DIRECT_READY|" + entityName + "|" + pgPoint);
    System.out.println("VON_MISES_SURFACE_DIRECT_READY|" + entityName + "|" + pgSurface);
  }

  public static void main(String[] args) throws Exception {
    Model model;
    try {
      model = ModelUtil.load("Model", MODEL_PATH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model: " + MODEL_PATH, e);
    }

    for (String[] ent : ENTITIES) {
      String entityName = ent[0];
      String shortTag = ent[1];
      String preflightBdf = ent[2];
      String solverBdf = ent[3];

      System.out.println("ENTITY_START|" + entityName);
      preflightOpen(model, entityName, preflightBdf);

      try {
        loadBdfForSolve(model, solverBdf);
        configureMr5AndLoads(model);
        model.study("std_mr5").run();
        buildVonMisesPlots(model, shortTag, entityName);
        exportVonMisesData(model, shortTag);
        System.out.println("ENTITY_DONE|" + entityName + "|ok=true");
      } catch (Exception e) {
        System.out.println("ENTITY_DONE|" + entityName + "|ok=false|" + e.getMessage());
      }
      try {
        model.save(MODEL_PATH);
        System.out.println("CHECKPOINT_SAVE|" + entityName + "|ok=true");
      } catch (Exception e) {
        System.out.println("CHECKPOINT_SAVE|" + entityName + "|ok=false|" + e.getMessage());
      }
    }

    try {
      model.save(MODEL_PATH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save model: " + MODEL_PATH, e);
    }
    System.out.println("Saved: " + MODEL_PATH);
  }
}

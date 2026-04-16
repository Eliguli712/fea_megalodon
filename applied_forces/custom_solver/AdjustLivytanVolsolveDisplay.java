import com.comsol.model.*;
import com.comsol.model.util.*;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class AdjustLivytanVolsolveDisplay {
  private static final String MPH =
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

  private static void safeSetResult(Model m, String pg, String key, String value) {
    try {
      m.result(pg).set(key, value);
    } catch (Exception ignored) {
    }
  }

  private static void safeSetResultFeature(Model m, String pg, String ft, String key, String value) {
    try {
      m.result(pg).feature(ft).set(key, value);
    } catch (Exception ignored) {
    }
  }

  private static String pickDataset(Model m, String preferred, String fallback) {
    try {
      String[] ds = m.result().dataset().tags();
      for (String d : ds) {
        if (preferred.equals(d)) return d;
      }
      for (String d : ds) {
        if (fallback.equals(d)) return d;
      }
      if (ds.length > 0) return ds[0];
    } catch (Exception ignored) {
    }
    return null;
  }

  private static void ensurePlotGroup(Model m, String pg, String label, String dset, String expr, String unit) {
    if (!hasResult(m, pg)) {
      m.result().create(pg, "PlotGroup3D");
    }
    try {
      m.result(pg).label(label);
    } catch (Exception ignored) {
    }

    if (dset != null && !dset.isEmpty()) {
      safeSetResult(m, pg, "data", dset);
    }

    // Bind to component view for stable opening in GUI.
    safeSetResult(m, pg, "view", "view1");
    safeSetResult(m, pg, "window", "window1");

    if (!hasResultFeature(m, pg, "surf1")) {
      m.result(pg).create("surf1", "Surface");
    }

    safeSetResultFeature(m, pg, "surf1", "expr", expr);
    if (unit != null && !unit.isEmpty()) {
      safeSetResultFeature(m, pg, "surf1", "unit", unit);
    }
    safeSetResultFeature(m, pg, "surf1", "descr", label);
    safeSetResultFeature(m, pg, "surf1", "resolution", "normal");
    safeSetResultFeature(m, pg, "surf1", "colortable", "ThermalLight");

    try {
      m.result(pg).feature("surf1").selection().all();
    } catch (Exception ignored) {
    }

    try {
      m.result(pg).run();
    } catch (Exception e) {
      p("PLOT_RUN_WARN|" + pg + "|" + e.getMessage());
    }
  }

  private static void patchCompView(Model m) {
    try {
      String[] views = m.component("comp1").view().tags();
      p("VIEWS_BEFORE|" + String.join(",", views));
    } catch (Exception e) {
      p("VIEWS_BEFORE|<unknown>");
    }

    // Common view tag in component models.
    try {
      m.component("comp1").view("view1");
    } catch (Exception e) {
      try {
        m.component("comp1").view().create("view1", 3);
      } catch (Exception ignored) {
      }
    }

    // Set conservative display properties; unsupported keys are ignored.
    try { m.component("comp1").view("view1").set("locked", "off"); } catch (Exception ignored) {}
    try { m.component("comp1").view("view1").set("showgrid", "off"); } catch (Exception ignored) {}
    try { m.component("comp1").view("view1").set("showaxis", "off"); } catch (Exception ignored) {}
    try { m.component("comp1").view("view1").set("projection", "perspective"); } catch (Exception ignored) {}
    try { m.component("comp1").view("view1").set("rendermesh", "on"); } catch (Exception ignored) {}
    try { m.component("comp1").view("view1").set("transparency", "off"); } catch (Exception ignored) {}

    // If available, fit extents in the current view.
    try {
      String[] views = m.component("comp1").view().tags();
      p("VIEWS_AFTER|" + String.join(",", views));
    } catch (Exception e) {
      p("VIEWS_AFTER|<unknown>");
    }
  }

  public static void main(String[] args) {
    Model m;
    try {
      m = ModelUtil.load("Model", MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load model", e);
    }

    String backup = MPH.replace(
        ".mph",
        ".pre_displayfix_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + ".mph"
    );

    try {
      m.save(backup);
      p("BACKUP|" + backup);
    } catch (IOException e) {
      p("BACKUP_WARN|" + e.getMessage());
    }

    String dsetGeom = pickDataset(m, "dset1", "dset2");
    String dsetVms = pickDataset(m, "dset2", "dset1");
    if (dsetVms == null) dsetVms = dsetGeom;

    p("DATASET_GEOM|" + dsetGeom);
    p("DATASET_VMS|" + dsetVms);

    patchCompView(m);

    ensurePlotGroup(m, "pg_geom_preview", "Livytan Geometry Preview", dsetGeom, "1", "1");
    ensurePlotGroup(m, "pg_vms_preview", "Livytan Von Mises Preview", dsetVms, "solid.mises", "Pa");

    // Keep pg1 valid and visible for default opening in some COMSOL layouts.
    if (!hasResult(m, "pg1")) {
      m.result().create("pg1", "PlotGroup3D");
    }
    ensurePlotGroup(m, "pg1", "Livytan Default Geometry View", dsetGeom, "1", "1");

    try {
      String[] pgs = m.result().tags();
      p("PLOT_GROUPS_AFTER|" + String.join(",", pgs));
    } catch (Exception e) {
      p("PLOT_GROUPS_AFTER|<unknown>");
    }

    try {
      int nv = m.component("comp1").mesh("mesh1").getNumVertex();
      int nt = m.component("comp1").mesh("mesh1").getNumElem("tri");
      int ne = m.component("comp1").mesh("mesh1").getNumElem("tet");
      p("MESH_COUNTS|vertices=" + nv + "|tri=" + nt + "|tet=" + ne);
    } catch (Exception e) {
      p("MESH_COUNTS|<unavailable>|" + e.getMessage());
    }

    try {
      m.save(MPH);
      p("SAVED|" + MPH);
    } catch (IOException e) {
      throw new RuntimeException("Failed to save model", e);
    }
  }
}

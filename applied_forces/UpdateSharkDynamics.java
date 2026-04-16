import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class UpdateSharkDynamics {
  private static boolean physicsExists(Model model, String compTag, String physTag) {
    try {
      model.component(compTag).physics(physTag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean selectionExists(Model model, String compTag, String selTag) {
    try {
      model.component(compTag).selection(selTag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean studyExists(Model model, String studyTag) {
    try {
      model.study(studyTag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean studyStepExists(Model model, String studyTag, String stepTag) {
    try {
      model.study(studyTag).feature(stepTag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  private static boolean physicsFeatureExists(Model model, String compTag, String physTag, String featTag) {
    try {
      model.component(compTag).physics(physTag).feature(featTag);
      return true;
    } catch (Exception e) {
      return false;
    }
  }

  public static Model run() {
    String mphPath = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";

    Model model;
    try {
      model = ModelUtil.load("Model", mphPath);
    } catch (IOException e) {
      throw new RuntimeException("Failed to load MPH file: " + mphPath, e);
    }

    // Thrust and snout-control parameters
    model.param().set("geom_unit", "m");
    model.param().descr("geom_unit", "Interpret STL coordinates as meters (full-body shark scale).");
    model.param().set("import_scale", "1");
    model.param().descr("import_scale", "Import scaling multiplier.");
    model.param().set("Lx", "10.1089[m]");
    model.param().descr("Lx", "Bounding-box size in X.");
    model.param().set("Ly", "7.0574[m]");
    model.param().descr("Ly", "Bounding-box size in Y.");
    model.param().set("Lz", "25.2196[m]");
    model.param().descr("Lz", "Bounding-box size in Z.");
    model.param().set("Ldiag", "28.0718[m]");
    model.param().descr("Ldiag", "Bounding-box diagonal (global characteristic length).");
    model.param().set("edge_med", "0.1652[m]");
    model.param().descr("edge_med", "Median triangle edge length.");
    model.param().set("edge_p90", "0.3551[m]");
    model.param().descr("edge_p90", "90th percentile triangle edge length.");
    model.param().set("repair_tol", "0.003[m]");
    model.param().descr("repair_tol", "Stitch/repair tolerance (~1e-4 * Ldiag).");
    model.param().set("merge_vertex_tol", "0.010[m]");
    model.param().descr("merge_vertex_tol", "Merge near-duplicate vertices.");
    model.param().set("keep_largest_component", "1");
    model.param().descr("keep_largest_component", "Keep main shark body component.");
    model.param().set("remove_small_component_area", "0.50[m^2]");
    model.param().descr("remove_small_component_area", "Remove tiny disconnected artifacts.");
    model.param().set("close_hole_max", "0.30[m]");
    model.param().descr("close_hole_max", "Close only small holes; avoid aggressive capping.");
    model.param().set("hmax_surf", "0.40[m]");
    model.param().descr("hmax_surf", "Maximum surface mesh size.");
    model.param().set("hmin_surf", "0.05[m]");
    model.param().descr("hmin_surf", "Minimum surface mesh size.");
    model.param().set("growth_surf", "1.20");
    model.param().descr("growth_surf", "Surface mesh growth rate.");
    model.param().set("curv_factor_surf", "0.35");
    model.param().descr("curv_factor_surf", "Curvature refinement factor (surface mesh).");
    model.param().set("narrow_res_surf", "0.70");
    model.param().descr("narrow_res_surf", "Narrow-region resolution factor.");
    model.param().set("hmax_vol", "0.60[m]");
    model.param().descr("hmax_vol", "Maximum volume mesh size.");
    model.param().set("hmin_vol", "0.06[m]");
    model.param().descr("hmin_vol", "Minimum volume mesh size.");
    model.param().set("growth_vol", "1.25");
    model.param().descr("growth_vol", "Volume mesh growth rate.");
    model.param().set("curv_factor_vol", "0.40");
    model.param().descr("curv_factor_vol", "Curvature refinement factor (volume mesh).");

    model.param().set("thrust_load", "5e4[Pa]");
    model.param().descr("thrust_load", "Equivalent snout thrust traction magnitude.");

    model.param().set("snout_frac", "0.05");
    model.param().descr("snout_frac", "Fraction of body length near the snout used for load selection.");

    model.param().set("snout_zmin", "Lz*(1-snout_frac)");
    model.param().descr("snout_zmin", "Snout selection lower Z bound.");

    model.param().set("snout_zmax", "Lz");
    model.param().descr("snout_zmax", "Snout selection upper Z bound.");

    // Create/overwrite snout selection box on boundaries (geom dim = 2)
    if (!selectionExists(model, "comp1", "sel_snout")) {
      model.component("comp1").selection().create("sel_snout", "Box");
    }
    model.component("comp1").selection("sel_snout").label("Snout selection");
    model.component("comp1").selection("sel_snout").set("entitydim", "2");
    model.component("comp1").selection("sel_snout").set("xmin", "0[m]");
    model.component("comp1").selection("sel_snout").set("xmax", "Lx");
    model.component("comp1").selection("sel_snout").set("ymin", "0[m]");
    model.component("comp1").selection("sel_snout").set("ymax", "Ly");
    model.component("comp1").selection("sel_snout").set("zmin", "snout_zmin");
    model.component("comp1").selection("sel_snout").set("zmax", "snout_zmax");

    // Solid Mechanics physics
    if (!physicsExists(model, "comp1", "solid")) {
      model.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
    }
    model.component("comp1").physics("solid").label("Solid Mechanics");

    // Boundary traction/pressure as thrust on snout.
    // Try known feature IDs in order; continue even if one fails.
    String loadTag = "bndl1";
    if (!physicsFeatureExists(model, "comp1", "solid", loadTag)) {
      String[] candidateOps = new String[] {"BoundaryLoad", "Pressure"};
      boolean created = false;
      for (String op : candidateOps) {
        try {
          model.component("comp1").physics("solid").create(loadTag, op, 2);
          created = true;
          break;
        } catch (Exception ignored) {
          // try next operation ID
        }
      }
      if (!created) {
        throw new RuntimeException("Could not create snout load feature with supported IDs.");
      }
    }

    model.component("comp1").physics("solid").feature(loadTag).label("Snout thrust load");
    model.component("comp1").physics("solid").feature(loadTag).selection().named("sel_snout");

    // Set load expression with tolerant fallbacks across feature types.
    boolean loadSet = false;
    try {
      model.component("comp1").physics("solid").feature(loadTag).set("LoadType", "ForcePerArea");
      model.component("comp1").physics("solid").feature(loadTag).set("FperArea", new String[] {"0", "0", "thrust_load"});
      loadSet = true;
    } catch (Exception ignored) {}
    if (!loadSet) {
      try {
        model.component("comp1").physics("solid").feature(loadTag).set("p0", "-thrust_load");
        loadSet = true;
      } catch (Exception ignored) {}
    }
    if (!loadSet) {
      try {
        model.component("comp1").physics("solid").feature(loadTag).set("Ftot", new String[] {"0", "0", "thrust_load"});
        loadSet = true;
      } catch (Exception ignored) {}
    }

    // Solid mechanics study
    if (!studyExists(model, "std1")) {
      model.study().create("std1");
    }
    model.study("std1").label("Solid Mechanics Study");

    if (!studyStepExists(model, "std1", "stat")) {
      model.study("std1").create("stat", "Stationary");
    }
    model.study("std1").feature("stat").activate("solid", true);

    return model;
  }

  public static void main(String[] args) {
    run();
  }
}

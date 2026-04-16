import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class TryImportMshSolve {
  private static void log(String s) { System.out.println(s); }
  private static boolean hasFeature(Model m, String c, String p, String f) {
    try { m.component(c).physics(p).feature(f); return true; } catch (Exception e) { return false; }
  }
  public static Model run() {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics_importmsh_test.mph";
    String msh = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol.bdf";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    // First, create a tetra volume mesh in the mesh part sequence (mpart1).
    try {
      try { m.mesh("mpart1").feature().remove("ftet_auto"); } catch (Exception e) {}
      try { m.mesh("mpart1").feature().remove("size_auto"); } catch (Exception e) {}
      m.mesh("mpart1").feature().create("size_auto", "Size");
      try {
        m.mesh("mpart1").feature("size_auto").set("hauto", 4);
        m.mesh("mpart1").feature("size_auto").set("hmax", "hmax_vol");
        m.mesh("mpart1").feature("size_auto").set("hmin", "hmin_vol");
        m.mesh("mpart1").feature("size_auto").set("hgrad", "growth_vol");
        m.mesh("mpart1").feature("size_auto").set("curv", "curv_factor_vol");
      } catch (Exception e) {}
      m.mesh("mpart1").feature().create("ftet_auto", "FreeTet");
      m.mesh("mpart1").run();
      log("mpart1 tetra meshing succeeded");
    } catch (Exception e) {
      log("mpart1 tetra meshing failed: " + e.getMessage());
      e.printStackTrace();
    }

    // Import from mpart1 sequence into comp1 mesh sequence.
    try {
      try { m.component("comp1").mesh("mesh1").feature().remove("impmsh"); } catch (Exception e) {}
      m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "sequence");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("sequence", "mpart1");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("buildsource", "on");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("domelemsequence", "on");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("unmesheddom", "on");
      try { m.component("comp1").mesh("mesh1").feature().remove("size_tet"); } catch (Exception e) {}
      try { m.component("comp1").mesh("mesh1").feature().remove("ftet_tet"); } catch (Exception e) {}
      m.component("comp1").mesh("mesh1").feature().create("size_tet", "Size");
      try {
        m.component("comp1").mesh("mesh1").feature("size_tet").set("hauto", 5);
      } catch (Exception e) {}
      m.component("comp1").mesh("mesh1").feature().create("ftet_tet", "FreeTet");
      m.component("comp1").mesh("mesh1").run();
      log("mesh import+run succeeded");
    } catch (Exception e) {
      log("mesh import failed: " + e.getMessage());
      e.printStackTrace();
    }

    // Domain: all imported domains
    try { m.component("comp1").physics("solid").selection().all(); log("solid domain set all"); }
    catch (Exception e) { log("solid selection all failed: " + e.getMessage()); }

    // Tail fixed boundary selection by low z (rear support)
    try { m.component("comp1").selection().create("sel_tail_fix", "Box"); } catch (Exception e) {}
    try {
      m.component("comp1").selection("sel_tail_fix").set("entitydim", "2");
      m.component("comp1").selection("sel_tail_fix").set("xmin", "-1e9[m]");
      m.component("comp1").selection("sel_tail_fix").set("xmax", "1e9[m]");
      m.component("comp1").selection("sel_tail_fix").set("ymin", "-1e9[m]");
      m.component("comp1").selection("sel_tail_fix").set("ymax", "1e9[m]");
      m.component("comp1").selection("sel_tail_fix").set("zmin", "-1e9[m]");
      m.component("comp1").selection("sel_tail_fix").set("zmax", "4[m]");
    } catch (Exception e) { log("tail selection configure failed: " + e.getMessage()); }

    if (!hasFeature(m, "comp1", "solid", "fix1")) {
      try { m.component("comp1").physics("solid").create("fix1", "Fixed", 2); } catch (Exception e) {}
    }
    try {
      m.component("comp1").physics("solid").feature("fix1").selection().named("sel_tail_fix");
      log("fixed boundary set");
    } catch (Exception e) {
      log("fixed boundary failed: " + e.getMessage());
    }

    // Snout load region by high z
    try { m.component("comp1").selection().create("sel_snout2", "Box"); } catch (Exception e) {}
    try {
      m.component("comp1").selection("sel_snout2").set("entitydim", "2");
      m.component("comp1").selection("sel_snout2").set("xmin", "-1e9[m]");
      m.component("comp1").selection("sel_snout2").set("xmax", "1e9[m]");
      m.component("comp1").selection("sel_snout2").set("ymin", "-1e9[m]");
      m.component("comp1").selection("sel_snout2").set("ymax", "1e9[m]");
      m.component("comp1").selection("sel_snout2").set("zmin", "20[m]");
      m.component("comp1").selection("sel_snout2").set("zmax", "1e9[m]");
    } catch (Exception e) { log("snout selection configure failed: " + e.getMessage()); }

    try {
      m.component("comp1").physics("solid").feature("bndl1").selection().named("sel_snout2");
      m.component("comp1").physics("solid").feature("bndl1").set("forceType", "ForceArea");
      m.component("comp1").physics("solid").feature("bndl1").set("force_src", "userdef");
      m.component("comp1").physics("solid").feature("bndl1").set("force", new String[]{"0","0","thrust_load"});
      m.component("comp1").physics("solid").feature("bndl1").active(true);
      try { m.component("comp1").physics("solid").feature("bndl_pr").active(false); } catch (Exception ex) {}
      log("snout thrust configured");
    } catch (Exception e) {
      log("snout load failed: " + e.getMessage());
    }

    // Use Neo-Hookean setup and run
    try {
      try { m.component("comp1").physics("solid").feature("hmm_nh").active(true); } catch (Exception e) {}
      try { m.component("comp1").physics("solid").feature("hmm_og").active(false); } catch (Exception e) {}
      try { m.component("comp1").physics("solid").feature("hmm_mr2").active(false); } catch (Exception e) {}
      try { m.component("comp1").physics("solid").feature("hmm_mr5").active(false); } catch (Exception e) {}
    } catch (Exception e) { log("hyperelastic activation failed: " + e.getMessage()); }

    try { m.study("std1").run(); log("std1 run complete"); }
    catch (Exception e) { log("std1 run failed: " + e.getMessage()); e.printStackTrace(); }

    try {
      try { m.result().numerical().remove("max_vms_test"); } catch (Exception e) {}
      m.result().numerical().create("max_vms_test", "MaxVolume");
      m.result().numerical("max_vms_test").set("expr", new String[]{"solid.mises"});
      m.result().numerical("max_vms_test").set("unit", new String[]{"Pa"});
      m.result().numerical("max_vms_test").setResult();
      double[][] v = m.result().numerical("max_vms_test").getReal();
      if (v != null && v.length > 0 && v[0].length > 0) log("max solid.mises = " + v[0][0]);
      else log("max solid.mises unavailable");
    } catch (Exception e) { log("max eval failed: " + e.getMessage()); }

    try {
      try { m.result().remove("pg_vms_test"); } catch (Exception e) {}
      m.result().create("pg_vms_test", "PlotGroup3D");
      m.result("pg_vms_test").set("data", "dset1");
      m.result("pg_vms_test").create("surf1", "Surface");
      m.result("pg_vms_test").feature("surf1").set("expr", "solid.mises");
      m.result("pg_vms_test").run();
      log("vms plot generated");
    } catch (Exception e) { log("vms plot failed: " + e.getMessage()); }

    try { m.save(out); }
    catch (IOException e) { throw new RuntimeException(e); }
    return m;
  }

  public static void main(String[] args) { run(); }
}

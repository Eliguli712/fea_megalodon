import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class TryComp2FromSTL {
  private static boolean compExists(Model m, String c) { try { m.component(c); return true; } catch (Exception e) { return false; } }
  private static boolean studyExists(Model m, String s) { try { m.study(s); return true; } catch (Exception e) { return false; } }

  private static void sset(Model m, String c, String p, String f, String k, String v) {
    try { m.component(c).physics(p).feature(f).set(k,v); } catch (Exception e) {}
  }

  public static Model run() {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics_comp2_test.mph";
    String stl = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface.stl";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    if (!compExists(m, "comp2")) m.component().create("comp2", true);

    // Fresh geom/mesh in comp2
    try { m.component("comp2").geom().create("geom1", 3); } catch (Exception e) {}
    try { m.component("comp2").mesh().create("mesh1"); } catch (Exception e) {}

    try {
      m.component("comp2").geom("geom1").create("imp1", "Import");
    } catch (Exception e) {
      System.out.println("geom import create failed: " + e.getMessage());
    }

    try { m.component("comp2").geom("geom1").feature("imp1").set("filename", stl); } catch (Exception e) {}
    try { m.component("comp2").geom("geom1").run(); } catch (Exception e) { System.out.println("geom run failed: " + e.getMessage()); }

    try { m.component("comp2").mesh("mesh1").automatic(true); } catch (Exception e) {}
    try { m.component("comp2").mesh("mesh1").run(); } catch (Exception e) { System.out.println("mesh run failed: " + e.getMessage()); }

    try { m.component("comp2").selection().create("sel_fix", "Box"); } catch (Exception e) {}
    try {
      m.component("comp2").selection("sel_fix").set("entitydim", "2");
      m.component("comp2").selection("sel_fix").set("zmin", "-1e9[m]");
      m.component("comp2").selection("sel_fix").set("zmax", "5[m]");
      m.component("comp2").selection("sel_fix").set("xmin", "-1e9[m]");
      m.component("comp2").selection("sel_fix").set("xmax", "1e9[m]");
      m.component("comp2").selection("sel_fix").set("ymin", "-1e9[m]");
      m.component("comp2").selection("sel_fix").set("ymax", "1e9[m]");
    } catch (Exception e) {}

    try { m.component("comp2").selection().create("sel_load", "Box"); } catch (Exception e) {}
    try {
      m.component("comp2").selection("sel_load").set("entitydim", "2");
      m.component("comp2").selection("sel_load").set("zmin", "20[m]");
      m.component("comp2").selection("sel_load").set("zmax", "1e9[m]");
      m.component("comp2").selection("sel_load").set("xmin", "-1e9[m]");
      m.component("comp2").selection("sel_load").set("xmax", "1e9[m]");
      m.component("comp2").selection("sel_load").set("ymin", "-1e9[m]");
      m.component("comp2").selection("sel_load").set("ymax", "1e9[m]");
    } catch (Exception e) {}

    try { m.component("comp2").physics().create("solid2", "SolidMechanics", "geom1"); } catch (Exception e) {}
    sset(m, "comp2", "solid2", "lemm1", "E_mat", "userdef");
    sset(m, "comp2", "solid2", "lemm1", "E", "1e8[Pa]");
    sset(m, "comp2", "solid2", "lemm1", "nu_mat", "userdef");
    sset(m, "comp2", "solid2", "lemm1", "nu", "0.3");
    sset(m, "comp2", "solid2", "lemm1", "rho_mat", "userdef");
    sset(m, "comp2", "solid2", "lemm1", "rho", "1100[kg/m^3]");

    try { m.component("comp2").physics("solid2").create("fix1", "Fixed", 2); } catch (Exception e) {}
    try { m.component("comp2").physics("solid2").feature("fix1").selection().named("sel_fix"); } catch (Exception e) {}

    try { m.component("comp2").physics("solid2").create("bndl1", "BoundaryLoad", 2); } catch (Exception e) {}
    try { m.component("comp2").physics("solid2").feature("bndl1").selection().named("sel_load"); } catch (Exception e) {}
    sset(m, "comp2", "solid2", "bndl1", "forceType", "FollowerPressure");
    sset(m, "comp2", "solid2", "bndl1", "pressure", "2e4[Pa]");

    if (!studyExists(m, "std2")) m.study().create("std2");
    try { m.study("std2").create("stat", "Stationary"); } catch (Exception e) {}
    m.study("std2").feature("stat").activate("solid2", true);
    m.study("std2").label("Comp2 STL sanity");

    try { m.study("std2").run(); } catch (Exception e) { System.out.println("std2 failed: " + e.getMessage()); }

    try { m.result().create("pg2", "PlotGroup3D"); } catch (Exception e) {}
    try { m.result("pg2").create("surf1", "Surface"); } catch (Exception e) {}
    try { m.result("pg2").feature("surf1").set("expr", "solid2.mises"); } catch (Exception e) {}

    try {
      m.result().numerical().create("max2", "MaxSurface");
      m.result().numerical("max2").set("expr", new String[]{"solid2.mises"});
      m.result().numerical("max2").set("unit", new String[]{"Pa"});
      m.result().numerical("max2").set("descr", new String[]{"von Mises"});
      m.result().numerical("max2").set("data", "dset6");
      m.result().numerical("max2").setResult();
      double[][] v = m.result().numerical("max2").getReal();
      if (v != null && v.length > 0 && v[0].length > 0) {
        System.out.println("max solid2.mises = " + v[0][0]);
      } else {
        System.out.println("max solid2.mises unavailable");
      }
    } catch (Exception e) {
      System.out.println("max eval failed: " + e.getMessage());
    }

    try { m.save(out); } catch (IOException e) { throw new RuntimeException(e); }
    return m;
  }

  public static void main(String[] args) { run(); }
}

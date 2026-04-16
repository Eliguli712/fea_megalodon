import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class TryMeshComponentSolve {
  private static boolean physicsExists(Model m, String c, String p) {
    try { m.component(c).physics(p); return true; } catch (Exception e) { return false; }
  }
  private static boolean featExists(Model m, String c, String p, String f) {
    try { m.component(c).physics(p).feature(f); return true; } catch (Exception e) { return false; }
  }
  private static boolean studyExists(Model m, String s) {
    try { m.study(s); return true; } catch (Exception e) { return false; }
  }

  private static void sset(Model m, String c, String p, String f, String k, String v) {
    try { m.component(c).physics(p).feature(f).set(k,v); } catch (Exception e) {}
  }

  public static Model run() {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics_mesh_component_test.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    // Mesh component solid mechanics
    if (!physicsExists(m, "mcomp1", "solidmc")) {
      m.component("mcomp1").physics().create("solidmc", "SolidMechanics", "mgeom1");
    }
    m.component("mcomp1").physics("solidmc").label("Solid Mechanics (mesh component)");

    if (featExists(m, "mcomp1", "solidmc", "lemm1")) {
      sset(m, "mcomp1", "solidmc", "lemm1", "E_mat", "userdef");
      sset(m, "mcomp1", "solidmc", "lemm1", "E", "1e8[Pa]");
      sset(m, "mcomp1", "solidmc", "lemm1", "nu_mat", "userdef");
      sset(m, "mcomp1", "solidmc", "lemm1", "nu", "0.3");
      sset(m, "mcomp1", "solidmc", "lemm1", "rho_mat", "userdef");
      sset(m, "mcomp1", "solidmc", "lemm1", "rho", "1100[kg/m^3]");
    }

    if (!featExists(m, "mcomp1", "solidmc", "rms1")) {
      try { m.component("mcomp1").physics("solidmc").create("rms1", "RigidMotionSuppression", 3); }
      catch (Exception e) {}
    }

    if (!featExists(m, "mcomp1", "solidmc", "bndl1")) {
      m.component("mcomp1").physics("solidmc").create("bndl1", "BoundaryLoad", 2);
    }
    sset(m, "mcomp1", "solidmc", "bndl1", "forceType", "FollowerPressure");
    sset(m, "mcomp1", "solidmc", "bndl1", "pressure", "2e4[Pa]");

    if (!studyExists(m, "std_mc")) m.study().create("std_mc");
    m.study("std_mc").label("Mesh component sanity study");
    try { m.study("std_mc").create("stat", "Stationary"); } catch (Exception e) {}
    m.study("std_mc").feature("stat").activate("solidmc", true);

    try { m.study("std_mc").run(); }
    catch (Exception e) { System.out.println("std_mc failed: " + e.getMessage()); }

    try { m.save(out); }
    catch (IOException e) { throw new RuntimeException(e); }
    return m;
  }

  public static void main(String[] args) { run(); }
}

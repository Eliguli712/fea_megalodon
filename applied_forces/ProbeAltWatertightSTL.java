import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeAltWatertightSTL {
  private static void p(String s){System.out.println(s);}  
  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String stl = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_comsol_watertight.stl";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeAltWatertightSTL_Model.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); } catch (IOException e) { throw new RuntimeException(e); }

    try { m.geom("part1").feature().remove("tor1"); p("removed tor1"); } catch (Exception e) { p("remove tor1 skipped: "+e.getMessage()); }

    // Update mesh-part STL import to watertight file
    try {
      m.mesh("mpart1").feature("imp1").set("source", "stl");
      m.mesh("mpart1").feature("imp1").set("filename", stl);
      m.mesh("mpart1").feature("imp1").set("createdom", "on");
      m.mesh("mpart1").feature("imp1").set("facepartition", "minimal");
      m.mesh("mpart1").run();
      p("mpart1 run ok");
    } catch (Exception e) {
      p("mpart1 run failed: " + e.getMessage());
      e.printStackTrace();
    }

    // Reimport sequence into comp1 geometry/mesh1
    try { m.component("comp1").mesh("mesh1").feature().remove("impmsh"); } catch (Exception e) {}
    try {
      m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "sequence");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("sequence", "mpart1");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("buildsource", "on");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("domelemsequence", "on");
      m.component("comp1").mesh("mesh1").feature("impmsh").set("unmesheddom", "on");
      m.component("comp1").mesh("mesh1").run();
      p("mesh1 import run ok");
    } catch (Exception e) {
      p("mesh1 import failed: " + e.getMessage());
      e.printStackTrace();
    }

    // Try volume tetra on mesh2 without exclusions
    try { m.component("comp1").mesh().remove("mesh2"); } catch (Exception e) {}
    try {
      m.component("comp1").mesh().create("mesh2", "geom1");
      m.component("comp1").mesh("mesh2").automatic(false);
      m.component("comp1").mesh("mesh2").feature().create("size1", "Size");
      m.component("comp1").mesh("mesh2").feature("size1").set("hauto", 5);
      m.component("comp1").mesh("mesh2").feature().create("ftet1", "FreeTet");
      m.component("comp1").mesh("mesh2").run();
      p("mesh2 run ok");
    } catch (Exception e) {
      p("mesh2 run failed: " + e.getMessage());
      e.printStackTrace();
    }

    try {
      int[] d = m.component("comp1").geom("geom1").getNDomains() > 0 ? null : null;
      p("geom domains=" + m.component("comp1").geom("geom1").getNDomains());
    } catch (Exception e) {
      p("geom domain count failed: " + e.getMessage());
    }

    try { m.save(out); } catch (IOException e) { throw new RuntimeException(e); }
  }
}

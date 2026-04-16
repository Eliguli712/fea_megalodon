import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeNastranProps {
  private static void g(Model m, String key) {
    try { System.out.println(key + "=" + m.component("comp1").mesh("mesh1").feature("impmsh").getString(key)); }
    catch (Exception e) { System.out.println(key + "<err>=" + e.getMessage()); }
  }
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.component("comp1").mesh("mesh1").feature("impmsh"); }
    catch (Exception e) { m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import"); }

    g(m, "source");
    g(m, "sourceswitch");
    g(m, "filename");

    try { m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "nastran"); } catch (Exception e) { System.out.println("set source nastran err=" + e.getMessage()); }
    try { m.component("comp1").mesh("mesh1").feature("impmsh").set("sourceswitch", "nastran"); } catch (Exception e) { System.out.println("set sourceswitch nastran err=" + e.getMessage()); }
    try { m.component("comp1").mesh("mesh1").feature("impmsh").set("filename", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol.msh"); } catch (Exception e) { System.out.println("set filename err=" + e.getMessage()); }

    g(m, "source");
    g(m, "sourceswitch");
    g(m, "filename");

    try {
      String[] s = m.component("comp1").mesh("mesh1").feature("impmsh").getAllowedPropertyValues("sourceswitch");
      if (s == null) System.out.println("allowed sourceswitch = null");
      else for (String v : s) System.out.println("sourceswitch opt=" + v);
    } catch (Exception e) { System.out.println("allowed sourceswitch err=" + e.getMessage()); }
  }
}

import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeConfigureMeshFromMsh {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.geom("part1").feature().remove("tor1"); } catch (Exception e) {}
    try { m.geom("part1").inputParam().set("solid", "0"); } catch (Exception e) {}
    try { m.geom("part1").inputParam().set("endsolid", "0"); } catch (Exception e) {}

    try { m.component("comp1").mesh("mesh1").feature().remove("impmsh"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature().remove("fin"); } catch (Exception e) {}
    m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");

    try {
      String[] opts = m.component("comp1").mesh("mesh1").feature("impmsh").getAllowedPropertyValues("source");
      if (opts != null) for (String s : opts) System.out.println("source opt="+s);
    } catch (Exception e) { System.out.println("opt err=" + e.getMessage()); }

    m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "nastran");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("filename", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol_conforming.bdf");
    try { m.component("comp1").mesh("mesh1").feature("impmsh").set("domelem", "on"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature("impmsh").set("createdom", "on"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature("impmsh").set("linearelem", "on"); } catch (Exception e) {}

    try { System.out.println("source=" + m.component("comp1").mesh("mesh1").feature("impmsh").getString("source")); } catch (Exception e) { System.out.println("source err="+e.getMessage()); }
    try { System.out.println("filename=" + m.component("comp1").mesh("mesh1").feature("impmsh").getString("filename")); } catch (Exception e) { System.out.println("filename err="+e.getMessage()); }

    try {
      m.component("comp1").mesh("mesh1").run("impmsh");
      System.out.println("run impmsh ok");
    } catch (Exception e) {
      System.out.println("run impmsh failed: " + e.getMessage());
      e.printStackTrace();
    }
  }
}

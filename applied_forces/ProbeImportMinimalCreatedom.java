import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeImportMinimalCreatedom {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.component("comp1").mesh("mesh1").feature().remove("impmsh"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature().remove("fin"); } catch (Exception e) {}
    m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");

    m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "nastran");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("filename", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol_conforming.bdf");
    try { m.component("comp1").mesh("mesh1").feature("impmsh").set("createdom", "on"); } catch (Exception e) { System.out.println("createdom err="+e.getMessage()); }

    try { System.out.println("source=" + m.component("comp1").mesh("mesh1").feature("impmsh").getString("source")); } catch (Exception e) {}

    try {
      m.component("comp1").mesh("mesh1").run("impmsh");
      System.out.println("run ok");
      m.component("comp1").physics("solid").selection().all();
      int nd = m.component("comp1").physics("solid").selection().entities(3).length;
      int nb = m.component("comp1").physics("solid").selection().entities(2).length;
      int ne = m.component("comp1").physics("solid").selection().entities(1).length;
      System.out.println("counts dom=" + nd + " bnd=" + nb + " edge=" + ne);
    } catch (Exception e) {
      System.out.println("run failed: " + e.getMessage());
      e.printStackTrace();
    }
  }
}

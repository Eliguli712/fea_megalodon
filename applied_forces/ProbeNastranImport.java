import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeNastranImport {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try {
      m.component("comp1").mesh("mesh1").feature("impmsh");
    } catch (Exception e) {
      m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
    }

    m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "nastran");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("filename", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol.msh");
    try { m.component("comp1").mesh("mesh1").feature("impmsh").set("domelem", "on"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature("impmsh").set("createdom", "on"); } catch (Exception e) {}

    try {
      m.component("comp1").mesh("mesh1").run();
      System.out.println("mesh1 import nastran run ok");
      try {
        m.component("comp1").physics("solid").selection().all();
        int nd = m.component("comp1").physics("solid").selection().entities(3).length;
        System.out.println("domain count=" + nd);
      } catch (Exception e) {
        System.out.println("domain count read failed: " + e.getMessage());
      }
    } catch (Exception e) {
      System.out.println("mesh1 import nastran failed: " + e.getMessage());
      e.printStackTrace();
    }
  }
}

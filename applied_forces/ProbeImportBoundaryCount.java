import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeImportBoundaryCount {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.component("comp1").mesh("mesh1").feature().remove("impmsh"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature().remove("fin"); } catch (Exception e) {}
    m.component("comp1").mesh("mesh1").feature().create("impmsh", "Import");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("source", "nastran");
    m.component("comp1").mesh("mesh1").feature("impmsh").set("filename", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol_conforming.bdf");
    m.component("comp1").mesh("mesh1").run("impmsh");

    m.component("comp1").physics("solid").selection().all();
    int nd = m.component("comp1").physics("solid").selection().entities(3).length;

    try { m.component("comp1").physics("solid").feature("fix1"); }
    catch (Exception e) { m.component("comp1").physics("solid").create("fix1", "Fixed", 2); }
    m.component("comp1").physics("solid").feature("fix1").selection().geom("geom1",2);
    m.component("comp1").physics("solid").feature("fix1").selection().all();
    int nb = m.component("comp1").physics("solid").feature("fix1").selection().entities(2).length;

    try { m.component("comp1").physics("solid").feature().remove("fixe_probe"); } catch (Exception e) {}
    m.component("comp1").physics("solid").create("fixe_probe", "Fixed", 1);
    m.component("comp1").physics("solid").feature("fixe_probe").selection().geom("geom1",1);
    m.component("comp1").physics("solid").feature("fixe_probe").selection().all();
    int ne = m.component("comp1").physics("solid").feature("fixe_probe").selection().entities(1).length;

    System.out.println("domain count=" + nd);
    System.out.println("boundary count=" + nb);
    System.out.println("edge count=" + ne);

    try { m.study("std1").feature("stat").set("mesh", new String[][]{{"geom1","mesh1"}}); } catch (Exception e) {}
    try {
      m.study("std1").run();
      System.out.println("std1 run ok");
    } catch (Exception e) {
      System.out.println("std1 run failed: " + e.getMessage());
      e.printStackTrace();
    }
  }
}

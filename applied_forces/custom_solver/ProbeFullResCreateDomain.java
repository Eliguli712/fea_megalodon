import com.comsol.model.*;
import com.comsol.model.util.*;

public class ProbeFullResCreateDomain {
  private static final String BDF = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/MBIE_White_Shark_HQ.bdf";

  public static void main(String[] args) {
    Model m = ModelUtil.create("Model");
    m.modelNode().create("mod1");
    m.component().create("comp1", false);
    m.component("comp1").geom().create("geom1", 3);
    m.component("comp1").mesh().create("mesh1");

    m.component("comp1").mesh("mesh1").feature().create("imp1", "Import");
    m.component("comp1").mesh("mesh1").feature("imp1").set("filename", BDF);
    m.component("comp1").mesh("mesh1").feature("imp1").set("source", "nastran");
    try { m.component("comp1").mesh("mesh1").feature("imp1").set("linearelem", "on"); } catch (Exception ignored) {}
    try { m.component("comp1").mesh("mesh1").feature("imp1").set("domelem", "on"); } catch (Exception ignored) {}
    try { m.component("comp1").mesh("mesh1").feature("imp1").set("createdom", "on"); } catch (Exception ignored) {}

    System.out.println("source=" + m.component("comp1").mesh("mesh1").feature("imp1").getString("source"));
    try { System.out.println("domelem=" + m.component("comp1").mesh("mesh1").feature("imp1").getString("domelem")); } catch (Exception ignored) {}
    try { System.out.println("createdom=" + m.component("comp1").mesh("mesh1").feature("imp1").getString("createdom")); } catch (Exception ignored) {}

    try {
      m.component("comp1").mesh("mesh1").run("imp1");
      System.out.println("imp1 run ok");
    } catch (Exception e) {
      System.out.println("imp1 run failed: " + e.getMessage());
    }

    try { m.component("comp1").mesh("mesh1").run("fin"); } catch (Exception ignored) {}

    try { System.out.println("num tri=" + m.component("comp1").mesh("mesh1").getNumElem("tri")); } catch (Exception e) { System.out.println("num tri err=" + e.getMessage()); }
    try { System.out.println("num tet=" + m.component("comp1").mesh("mesh1").getNumElem("tet")); } catch (Exception e) { System.out.println("num tet err=" + e.getMessage()); }

    try {
      m.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
      m.component("comp1").physics("solid").selection().all();
      int dom = m.component("comp1").physics("solid").selection().entities(3).length;
      int bnd = m.component("comp1").physics("solid").selection().entities(2).length;
      System.out.println("solid selection dom=" + dom + " bnd=" + bnd);
    } catch (Exception e) {
      System.out.println("solid selection failed: " + e.getMessage());
    }

    try {
      m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/ProbeFullResCreateDomain_Model.mph");
      System.out.println("saved probe model");
    } catch (Exception e) {
      System.out.println("save failed: " + e.getMessage());
    }
  }
}

import com.comsol.model.*;
import com.comsol.model.util.*;

public class ProbeObjGeomImport {
  private static final String OBJ =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth.obj";

  private static void p(String s) { System.out.println(s); }

  public static void main(String[] args) {
    Model m = ModelUtil.create("Model");
    m.modelNode().create("mod1");
    m.component().create("comp1", false);
    m.component("comp1").geom().create("geom1", 3);
    m.component("comp1").geom("geom1").create("imp1", "Import");
    m.component("comp1").geom("geom1").feature("imp1").set("filename", OBJ);

    try {
      m.component("comp1").geom("geom1").run();
      p("OBJ_IMPORT|ok");
      p("COUNTS|dom=" + m.component("comp1").geom("geom1").getNDomains()
          + "|bnd=" + m.component("comp1").geom("geom1").getNBoundaries()
          + "|edg=" + m.component("comp1").geom("geom1").getNEdges());
    } catch (Exception e) {
      p("OBJ_IMPORT|fail|" + e.getMessage());
      e.printStackTrace();
    }

    ModelUtil.disconnect();
  }
}

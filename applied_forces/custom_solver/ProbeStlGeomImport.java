import com.comsol.model.*;
import com.comsol.model.util.*;

public class ProbeStlGeomImport {
  private static final String STL =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth.stl";

  private static void p(String s) { System.out.println(s); }

  public static void main(String[] args) {
    Model m = ModelUtil.create("Model");
    m.modelNode().create("mod1");
    m.component().create("comp1", false);
    m.component("comp1").geom().create("geom1", 3);
    m.component("comp1").geom("geom1").create("imp1", "Import");
    m.component("comp1").geom("geom1").feature("imp1").set("filename", STL);
    try { m.component("comp1").geom("geom1").feature("imp1").set("facepartition", "minimal"); } catch (Exception ignored) {}

    try {
      m.component("comp1").geom("geom1").run();
      p("STL_IMPORT|ok");
      p("COUNTS|dom=" + m.component("comp1").geom("geom1").getNDomains()
          + "|bnd=" + m.component("comp1").geom("geom1").getNBoundaries()
          + "|edg=" + m.component("comp1").geom("geom1").getNEdges());
      p("FEATS|" + String.join(",", m.component("comp1").geom("geom1").feature().tags()));
    } catch (Exception e) {
      p("STL_IMPORT|fail|" + e.getMessage());
      e.printStackTrace();
    }

    ModelUtil.disconnect();
  }
}

import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeLivytanGeomState {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";

  private static void p(String s) { System.out.println(s); }

  private static void listTags(String label, String[] tags) {
    StringBuilder sb = new StringBuilder();
    sb.append(label).append("|");
    if (tags == null || tags.length == 0) {
      sb.append("<none>");
    } else {
      for (int i = 0; i < tags.length; i++) {
        if (i > 0) sb.append(",");
        sb.append(tags[i]);
      }
    }
    p(sb.toString());
  }

  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", MPH); }
    catch (IOException e) { throw new RuntimeException(e); }

    listTags("COMP", m.component().tags());
    try {
      listTags("GEOM", m.component("comp1").geom().tags());
      listTags("GEOM1_FEATS", m.component("comp1").geom("geom1").feature().tags());
      p("GEOM1_STATS|dom=" + m.component("comp1").geom("geom1").getNDomains()
          + "|bnd=" + m.component("comp1").geom("geom1").getNBoundaries()
          + "|edg=" + m.component("comp1").geom("geom1").getNEdges());
    } catch (Exception e) {
      p("GEOM_ERR|" + e.getMessage());
    }

    try {
      listTags("MESH", m.component("comp1").mesh().tags());
      listTags("MESH1_FEATS", m.component("comp1").mesh("mesh1").feature().tags());
      p("MESH1_COUNTS|v=" + m.component("comp1").mesh("mesh1").getNumVertex()
          + "|tri=" + m.component("comp1").mesh("mesh1").getNumElem("tri")
          + "|tet=" + m.component("comp1").mesh("mesh1").getNumElem("tet"));
    } catch (Exception e) {
      p("MESH_ERR|" + e.getMessage());
    }

    try {
      listTags("PHYS", m.component("comp1").physics().tags());
      int nd = m.component("comp1").physics("solid").selection().entities(3).length;
      p("SOLID_DOM_SEL|" + nd);
    } catch (Exception e) {
      p("PHYS_ERR|" + e.getMessage());
    }

    ModelUtil.disconnect();
  }
}

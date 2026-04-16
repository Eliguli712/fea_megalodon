import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeLivytanVisibilityCause {
  private static final String MPH =
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/custom_solver/livytan_melville_teeth_volsolve.mph";

  private static void p(String s){System.out.println(s);} 

  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", MPH); }
    catch (IOException e) { throw new RuntimeException(e); }

    p("COMPONENTS|" + String.join(",", m.component().tags()));

    try { p("COMP1_GEOMS|" + String.join(",", m.component("comp1").geom().tags())); }
    catch (Exception e) { p("COMP1_GEOMS|<none>|" + e.getMessage()); }

    try {
      String[] gf = m.component("comp1").geom("geom1").feature().tags();
      p("GEOM1_FEATURES|" + String.join(",", gf));
      m.component("comp1").geom("geom1").run();
      int nd = m.component("comp1").geom("geom1").getNDomains();
      int nb = m.component("comp1").geom("geom1").getNBoundaries();
      int ne = m.component("comp1").geom("geom1").getNEdges();
      p("GEOM_COUNTS|domains=" + nd + "|boundaries=" + nb + "|edges=" + ne);
    } catch (Exception e) {
      p("GEOM_COUNTS|<none>|" + e.getMessage());
    }

    try { p("MESH_FEATURES|" + String.join(",", m.component("comp1").mesh("mesh1").feature().tags())); }
    catch (Exception e) { p("MESH_FEATURES|<none>|" + e.getMessage()); }

    try {
      p("MESH_COUNTS|vertices=" + m.component("comp1").mesh("mesh1").getNumVertex()
       + "|tri=" + m.component("comp1").mesh("mesh1").getNumElem("tri")
       + "|tet=" + m.component("comp1").mesh("mesh1").getNumElem("tet"));
    } catch (Exception e) { p("MESH_COUNTS|<none>|" + e.getMessage()); }

    try { p("DATASETS|" + String.join(",", m.result().dataset().tags())); }
    catch (Exception e) { p("DATASETS|<none>|" + e.getMessage()); }

    try {
      for (String d : m.result().dataset().tags()) {
        String type=""; String sol="";
        try { type = m.result().dataset(d).getType(); } catch (Exception ignored) {}
        try { sol = m.result().dataset(d).getString("solution"); } catch (Exception ignored) {}
        p("DATASET|" + d + "|type=" + type + "|solution=" + sol);
      }
    } catch (Exception ignored) {}

    try { p("PLOT_GROUPS|" + String.join(",", m.result().tags())); }
    catch (Exception e) { p("PLOT_GROUPS|<none>|" + e.getMessage()); }

    String[] pgs = new String[]{"pg1","pg_geom_preview","pg_vms_preview"};
    for (String pg : pgs) {
      try {
        String data = "";
        try { data = m.result(pg).getString("data"); } catch (Exception ignored) {}
        String view = "";
        try { view = m.result(pg).getString("view"); } catch (Exception ignored) {}
        String expr = "";
        try { expr = m.result(pg).feature("surf1").getString("expr"); } catch (Exception ignored) {}
        p("PG|" + pg + "|data=" + data + "|view=" + view + "|expr=" + expr);
      } catch (Exception e) {
        p("PG|" + pg + "|<missing>");
      }
    }
  }
}

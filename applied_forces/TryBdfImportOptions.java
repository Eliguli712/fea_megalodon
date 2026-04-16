import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class TryBdfImportOptions {
  private static void tryCase(Model m, String linearelem, String domelem, String createdom, String facepartition) {
    String tag = "impopt";
    try { m.component("comp1").mesh("mesh1").feature().remove(tag); } catch (Exception e) {}
    m.component("comp1").mesh("mesh1").feature().create(tag, "Import");
    m.component("comp1").mesh("mesh1").feature(tag).set("source", "file");
    m.component("comp1").mesh("mesh1").feature(tag).set("filename", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol.bdf");
    try { m.component("comp1").mesh("mesh1").feature(tag).set("linearelem", linearelem); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature(tag).set("domelem", domelem); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature(tag).set("createdom", createdom); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature(tag).set("facepartition", facepartition); } catch (Exception e) {}

    String key = "linearelem=" + linearelem + " domelem=" + domelem + " createdom=" + createdom + " facepartition=" + facepartition;
    try {
      m.component("comp1").mesh("mesh1").run(tag);
      System.out.println("SUCCESS " + key);
      try {
        m.component("comp1").physics("solid").selection().all();
        int nd = m.component("comp1").physics("solid").selection().entities(3).length;
        int nb = m.component("comp1").physics("solid").selection().entities(2).length;
        int ne = m.component("comp1").physics("solid").selection().entities(1).length;
        System.out.println("counts dom=" + nd + " bnd=" + nb + " edge=" + ne);
      } catch (Exception e) {
        System.out.println("count read failed: " + e.getMessage());
      }
    } catch (Exception e) {
      System.out.println("FAIL " + key + " :: " + e.getMessage());
    }
  }

  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    String[] oo = new String[]{"on", "off"};
    for (String linearelem : oo) {
      for (String domelem : oo) {
        for (String createdom : oo) {
          for (String facepartition : new String[]{"auto", "off"}) {
            tryCase(m, linearelem, domelem, createdom, facepartition);
          }
        }
      }
    }
  }
}

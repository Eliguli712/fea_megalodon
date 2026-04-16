import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeFreshBdfGmshImport {
  private static void pr(Model m, String k) {
    try { System.out.println(k + "=" + m.component("comp1").mesh("mesh1").feature("imptest").getString(k)); }
    catch (Exception e) { System.out.println(k + "<err>=" + e.getMessage()); }
  }
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.component("comp1").mesh("mesh1").feature().remove("imptest"); } catch (Exception e) {}
    m.component("comp1").mesh("mesh1").feature().create("imptest", "Import");

    try { m.component("comp1").mesh("mesh1").feature("imptest").set("source", "file"); } catch (Exception e) { System.out.println("set source err=" + e.getMessage()); }
    try { m.component("comp1").mesh("mesh1").feature("imptest").set("filename", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol_gmsh.bdf"); } catch (Exception e) { System.out.println("set file err="+e.getMessage()); }
    pr(m, "source");
    pr(m, "filename");

    try { m.component("comp1").mesh("mesh1").run("imptest"); System.out.println("imptest run ok"); }
    catch (Exception e) { System.out.println("imptest run failed: " + e.getMessage()); e.printStackTrace(); }

    try {
      m.component("comp1").mesh("mesh1").feature().remove("impmsh");
    } catch (Exception e) {}
    m.component("comp1").mesh("mesh1").feature().duplicate("impmsh", "imptest");

    try { m.component("comp1").mesh("mesh1").run(); System.out.println("mesh1 run ok"); }
    catch (Exception e) { System.out.println("mesh1 run failed: " + e.getMessage()); e.printStackTrace(); }

    try {
      m.component("comp1").physics("solid").selection().all();
      int nd = m.component("comp1").physics("solid").selection().entities(3).length;
      int nb = m.component("comp1").physics("solid").selection().entities(2).length;
      int ne = m.component("comp1").physics("solid").selection().entities(1).length;
      System.out.println("counts dom=" + nd + " bnd=" + nb + " edge=" + ne);
    } catch (Exception e) {
      System.out.println("count read failed: " + e.getMessage());
    }
  }
}

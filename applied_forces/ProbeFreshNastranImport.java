import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeFreshNastranImport {
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

    try {
      String[] opts = m.component("comp1").mesh("mesh1").feature("imptest").getAllowedPropertyValues("source");
      if (opts!=null) for(String s:opts) System.out.println("source opt="+s);
    } catch (Exception e) { System.out.println("opts err="+e.getMessage()); }

    pr(m, "source");
    pr(m, "filename");

    try { m.component("comp1").mesh("mesh1").feature("imptest").set("source", "nastran"); } catch (Exception e) { System.out.println("set source err="+e.getMessage()); }
    try { m.component("comp1").mesh("mesh1").feature("imptest").set("filename", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol.msh"); } catch (Exception e) { System.out.println("set file err="+e.getMessage()); }
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

    try { m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ProbeFreshNastranImport_Model.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}

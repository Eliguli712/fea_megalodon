import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeCtetraOnlyImport2 {
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

    try { m.component("comp1").mesh("mesh1").feature("imptest").set("source", "nastran"); } catch (Exception e) { System.out.println("set source err="+e.getMessage()); }
    try { m.component("comp1").mesh("mesh1").feature("imptest").set("filename", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/__tracked_surface_tet_vol_ctetra_only.bdf"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature("imptest").set("domelem", "on"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature("imptest").set("createdom", "on"); } catch (Exception e) {}
    try { m.component("comp1").mesh("mesh1").feature("imptest").set("linearelem", "on"); } catch (Exception e) {}

    pr(m, "source");
    pr(m, "filename");

    try {
      m.component("comp1").mesh("mesh1").run("imptest");
      System.out.println("imptest run ok");
    } catch (Exception e) {
      System.out.println("imptest run failed: " + e.getMessage());
      e.printStackTrace();
    }
  }
}

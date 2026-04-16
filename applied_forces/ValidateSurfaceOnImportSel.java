import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ValidateSurfaceOnImportSel {
  public static void main(String[] args) {
    String in = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    Model m;
    try { m = ModelUtil.load("Model", in); }
    catch (IOException e) { throw new RuntimeException(e); }

    String bsel = "mesh1_impmsh_mpart1_imp1___tracked_surface_stl";

    try { m.study("std1").run(); } catch (Exception e) { System.out.println("std1 failed: " + e.getMessage()); }

    try { m.result().numerical().remove("mxsurf_imp"); } catch (Exception e) {}
    try {
      m.result().numerical().create("mxsurf_imp", "MaxSurface");
      m.result().numerical("mxsurf_imp").set("data", "dset6");
      m.result().numerical("mxsurf_imp").set("expr", new String[]{"solid.mises"});
      m.result().numerical("mxsurf_imp").set("unit", new String[]{"Pa"});
      m.result().numerical("mxsurf_imp").selection().named(bsel);
      m.result().numerical("mxsurf_imp").setResult();
      double[][] r = m.result().numerical("mxsurf_imp").getReal();
      double mx = (r!=null&&r.length>0&&r[0].length>0)?r[0][0]:Double.NaN;
      System.out.println("max_surface_mises_import_selection=" + mx);
    } catch (Exception e) {
      System.out.println("surface eval failed: " + e.getMessage());
    }

    try {
      m.result("pg_vms_std1").set("data", "dset6");
      m.result("pg_vms_std1").feature("surf1").selection().named(bsel);
      m.result("pg_vms_std1").run();
      System.out.println("pg_vms_std1(run with import selection)=ok");
    } catch (Exception e) {
      System.out.println("plot run failed: " + e.getMessage());
    }

    try { m.save("/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/ValidateSurfaceOnImportSel_Model.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }
  }
}

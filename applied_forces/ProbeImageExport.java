import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeImageExport {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.study("std1").run(); } catch (Exception e) { System.out.println("std1 failed: " + e.getMessage()); }
    try { m.result("pg_vms_std1").run(); } catch (Exception e) { System.out.println("plot run failed: " + e.getMessage()); }

    try { m.result().export().remove("imgtest"); } catch (Exception e) {}
    try {
      m.result().export().create("imgtest", "Image3D");
      String[] a = m.result().export("imgtest").getAllowedPropertyValues("imagetype");
      if (a != null) {
        System.out.println("imagetype opts=" + a.length);
        for (String s : a) System.out.println("  " + s);
      }
      m.result().export("imgtest").set("plotgroup", "pg_vms_std1");
      m.result().export("imgtest").set("pngfilename", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/exports/test_vms.png");
      m.result().export("imgtest").run();
      System.out.println("image export ok");
    } catch (Exception e) {
      System.out.println("image export failed: " + e.getMessage());
      e.printStackTrace();
    }
  }
}

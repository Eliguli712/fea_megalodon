import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ProbeImageExport2 {
  public static void main(String[] args) {
    Model m;
    try { m = ModelUtil.load("Model", "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph"); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.study("std1").run(); } catch (Exception e) {}
    try { m.result("pg_vms_std1").run(); } catch (Exception e) {}

    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/exports/test_vms2.png";
    try { m.result().export().remove("imgtest2"); } catch (Exception e) {}

    try {
      m.result().export().create("imgtest2", "Image");
      m.result().export("imgtest2").set("plotgroup", "pg_vms_std1");
      try { m.result().export("imgtest2").set("imagetype", "png"); } catch (Exception e) {}
      try { m.result().export("imgtest2").set("size", "manual"); } catch (Exception e) {}
      try { m.result().export("imgtest2").set("unit", "px"); } catch (Exception e) {}
      try { m.result().export("imgtest2").set("width", 1200); } catch (Exception e) {}
      try { m.result().export("imgtest2").set("height", 900); } catch (Exception e) {}
      try { m.result().export("imgtest2").set("antialias", "off"); } catch (Exception e) {}
      m.result().export("imgtest2").set("pngfilename", out);
      m.result().export("imgtest2").run();
      System.out.println("image2 export ok: " + out);
    } catch (Exception e) {
      System.out.println("image2 export failed: " + e.getMessage());
      e.printStackTrace();
    }
  }
}

import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class ExportStd1Contrast {
  public static void main(String[] args) throws Exception {
    String mph = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";
    String out = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/exports/holocastic_full_body_images/std1_von_mises_contrast.png";
    Model m;
    try { m = ModelUtil.load("Model", mph); }
    catch (IOException e) { throw new RuntimeException(e); }

    try { m.result().remove("pg_std1_contrast"); } catch (Exception e) {}
    m.result().create("pg_std1_contrast", "PlotGroup3D");
    m.result("pg_std1_contrast").set("data", "dset6");
    m.result("pg_std1_contrast").label("std1 Von Mises High-Contrast");
    m.result("pg_std1_contrast").create("surf1", "Surface");
    m.result("pg_std1_contrast").feature("surf1").set("expr", "solid.mises");
    m.result("pg_std1_contrast").feature("surf1").set("colortable", "Turbo");
    m.result("pg_std1_contrast").feature("surf1").set("rangecoloractive", "on");
    m.result("pg_std1_contrast").feature("surf1").set("rangecolormin", "0[Pa]");
    m.result("pg_std1_contrast").feature("surf1").set("rangecolormax", "21771.45804865712[Pa]");
    m.result("pg_std1_contrast").run();

    try { m.result().export().remove("img_std1_contrast"); } catch (Exception e) {}
    m.result().export().create("img_std1_contrast", "Image3D");
    m.result().export("img_std1_contrast").set("plotgroup", "pg_std1_contrast");
    try { m.result().export("img_std1_contrast").set("imagetype", "png"); } catch (Exception e) {}
    try { m.result().export("img_std1_contrast").set("qualitylevel", "95"); } catch (Exception e) {}
    try { m.result().export("img_std1_contrast").set("unit", "px"); } catch (Exception e) {}
    try { m.result().export("img_std1_contrast").set("width", 1400); } catch (Exception e) {}
    try { m.result().export("img_std1_contrast").set("height", 980); } catch (Exception e) {}
    m.result().export("img_std1_contrast").set("pngfilename", out);
    m.result().export("img_std1_contrast").run();
    System.out.println("WROTE " + out);
  }
}

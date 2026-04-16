import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.IOException;

public class AddMr5GraphImagesToModel {
  private static final String MPH = "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/shark_dynamics.mph";

  private static void upsertImagePlot(Model m, String pgTag, String label, String imagePath) {
    try { m.result().remove(pgTag); } catch (Exception ignored) {}
    m.result().create(pgTag, "PlotGroup2D");
    m.result(pgTag).label(label);
    try { m.result(pgTag).set("data", "none"); } catch (Exception ignored) {}
    m.result(pgTag).create("img1", "Image");
    ResultFeature img = m.result(pgTag).feature("img1");
    img.label(label + " Image");
    img.set("sourcetype", "user");
    img.set("filename", imagePath);
    try { img.set("mapping", "auto"); } catch (Exception ignored) {}
    try { img.set("heightmode", "fit"); } catch (Exception ignored) {}
    try { img.set("coordinterpretation", "pixels"); } catch (Exception ignored) {}
    try { img.set("planetype", "xy"); } catch (Exception ignored) {}
    try { img.set("preserveaspect", true); } catch (Exception ignored) {}
    m.result(pgTag).run();
    System.out.println("ADDED " + pgTag + " -> " + imagePath);
  }

  public static void main(String[] args) throws Exception {
    Model m;
    try { m = ModelUtil.load("Model", MPH); }
    catch (IOException e) { throw new RuntimeException("Failed to load model", e); }

    upsertImagePlot(
      m,
      "pg_mr5_trailing_force_img",
      "MR5 Trailing Edge Force Graph",
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/exports/holocastic_full_body_images/mr5_frontstress_trailing_force.png"
    );
    upsertImagePlot(
      m,
      "pg_mr5_max_impact_img",
      "MR5 Max Impact Graph",
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/exports/holocastic_full_body_images/mr5_frontstress_max_impact.png"
    );
    upsertImagePlot(
      m,
      "pg_mr5_von_mises_img",
      "MR5 Max Avg Von Mises Graph",
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/exports/holocastic_full_body_images/mr5_frontstress_von_mises.png"
    );
    upsertImagePlot(
      m,
      "pg_mr5_instant_impact_img",
      "MR5 Instantaneous Impact Graph",
      "/Users/eliguli712/DataStructure/numerical_analysis/FEA_meg/applied_forces/exports/holocastic_full_body_images/mr5_frontstress_instant_impact.png"
    );

    try { m.save(MPH); }
    catch (IOException e) { throw new RuntimeException("Failed to save model", e); }
    System.out.println("AddMr5GraphImagesToModel done");
  }
}

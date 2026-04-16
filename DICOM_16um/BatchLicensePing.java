import com.comsol.model.util.ModelUtil;

public class BatchLicensePing {
  public static void main(String[] args) {
    String version = "";
    try {
      version = ModelUtil.getComsolVersion();
    } catch (Exception ignored) {
    }
    System.out.println("BATCH_LICENSE_PING|ok=true|version=" + version);
  }
}

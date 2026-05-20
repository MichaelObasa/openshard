resource "google_compute_network" "harbourdocs_vpc" {
  name                    = "harbourdocs-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "api_subnet" {
  name          = "harbourdocs-api-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.harbourdocs_vpc.id
}

resource "google_compute_subnetwork" "processor_subnet" {
  name          = "harbourdocs-processor-subnet"
  ip_cidr_range = "10.0.2.0/24"
  region        = var.region
  network       = google_compute_network.harbourdocs_vpc.id
}

resource "google_compute_router" "harbourdocs_router" {
  name    = "harbourdocs-router"
  region  = var.region
  network = google_compute_network.harbourdocs_vpc.id
}

resource "google_compute_router_nat" "harbourdocs_nat" {
  name                               = "harbourdocs-nat"
  router                             = google_compute_router.harbourdocs_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

# Deliberate flaw: allows all ingress from internet — should be restricted to known CIDRs
resource "google_compute_firewall" "allow_https" {
  name    = "harbourdocs-allow-https"
  network = google_compute_network.harbourdocs_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }

  source_ranges = ["0.0.0.0/0"]
}

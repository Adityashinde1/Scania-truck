resource "azurerm_container_registry" "scania_acr" {
  name                = var.scania_acr_name
  resource_group_name = var.scania_resource_group_name
  location            = var.scania_resource_group_location
  sku                 = var.scania_acr_sku
}